//=== StackAddrEscapeChecker.cpp ----------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines stack address leak checker, which checks if an invalid
// stack address is stored into a global or heap location. See CERT DCL30-C.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace ento;

namespace {
class StackAddrEscapeChecker : public Checker< check::PreStmt<ReturnStmt>,
                                               check::EndFunction > {
  mutable std::unique_ptr<BuiltinBug> BT_stackleak;
  mutable std::unique_ptr<BuiltinBug> BT_returnstack;

public:
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;
  void checkEndFunction(CheckerContext &Ctx) const;
private:
  void EmitStackError(CheckerContext &C, const MemRegion *R,
                      const Expr *RetE) const;
  static SourceRange genName(raw_ostream &os, const MemRegion *R,
                             ASTContext &Ctx);
};
}

SourceRange StackAddrEscapeChecker::genName(raw_ostream &os, const MemRegion *R,
                                            ASTContext &Ctx) {
    // Get the base region, stripping away fields and elements.
  R = R->getBaseRegion();
  SourceManager &SM = Ctx.getSourceManager();
  SourceRange range;
  os << "Address of ";

  // Check if the region is a compound literal.
  if (const CompoundLiteralRegion* CR = dyn_cast<CompoundLiteralRegion>(R)) {
    const CompoundLiteralExpr *CL = CR->getLiteralExpr();
    os << "stack memory associated with a compound literal "
          "declared on line "
        << SM.getExpansionLineNumber(CL->getLocStart())
        << " returned to caller";
    range = CL->getSourceRange();
  }
  else if (const AllocaRegion* AR = dyn_cast<AllocaRegion>(R)) {
    const Expr *ARE = AR->getExpr();
    SourceLocation L = ARE->getLocStart();
    range = ARE->getSourceRange();
    os << "stack memory allocated by call to alloca() on line "
       << SM.getExpansionLineNumber(L);
  }
  else if (const BlockDataRegion *BR = dyn_cast<BlockDataRegion>(R)) {
    const BlockDecl *BD = BR->getCodeRegion()->getDecl();
    SourceLocation L = BD->getLocStart();
    range = BD->getSourceRange();
    os << "stack-allocated block declared on line "
       << SM.getExpansionLineNumber(L);
  }
  else if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    os << "stack memory associated with local variable '"
       << VR->getString() << '\'';
    range = VR->getDecl()->getSourceRange();
  }
  else if (const CXXTempObjectRegion *TOR = dyn_cast<CXXTempObjectRegion>(R)) {
    QualType Ty = TOR->getValueType().getLocalUnqualifiedType();
    os << "stack memory associated with temporary object of type '";
    Ty.print(os, Ctx.getPrintingPolicy());
    os << "'";
    range = TOR->getExpr()->getSourceRange();
  }
  else {
    llvm_unreachable("Invalid region in ReturnStackAddressChecker.");
  }

  return range;
}

void StackAddrEscapeChecker::EmitStackError(CheckerContext &C, const MemRegion *R,
                                          const Expr *RetE) const {
  ExplodedNode *N = C.generateErrorNode();

  if (!N)
    return;

  if (!BT_returnstack)
    BT_returnstack.reset(
        new BuiltinBug(this, "Return of address to stack-allocated memory"));

  // Generate a report for this bug.
  SmallString<512> buf;
  llvm::raw_svector_ostream os(buf);
  SourceRange range = genName(os, R, C.getASTContext());
  os << " returned to caller";
  auto report = llvm::make_unique<BugReport>(*BT_returnstack, os.str(), N);
  report->addRange(RetE->getSourceRange());
  if (range.isValid())
    report->addRange(range);

  C.emitReport(std::move(report));
}

void StackAddrEscapeChecker::checkPreStmt(const ReturnStmt *RS,
                                          CheckerContext &C) const {

  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
  RetE = RetE->IgnoreParens();

  const LocationContext *LCtx = C.getLocationContext();
  SVal V = C.getState()->getSVal(RetE, LCtx);
  const MemRegion *R = V.getAsRegion();

  if (!R)
    return;

  const StackSpaceRegion *SS =
    dyn_cast_or_null<StackSpaceRegion>(R->getMemorySpace());

  if (!SS)
    return;

  // Return stack memory in an ancestor stack frame is fine.
  const StackFrameContext *CurFrame = LCtx->getCurrentStackFrame();
  const StackFrameContext *MemFrame = SS->getStackFrame();
  if (MemFrame != CurFrame)
    return;

  // Automatic reference counting automatically copies blocks.
  if (C.getASTContext().getLangOpts().ObjCAutoRefCount &&
      isa<BlockDataRegion>(R))
    return;

  // Returning a record by value is fine. (In this case, the returned
  // expression will be a copy-constructor, possibly wrapped in an
  // ExprWithCleanups node.)
  if (const ExprWithCleanups *Cleanup = dyn_cast<ExprWithCleanups>(RetE))
    RetE = Cleanup->getSubExpr();
  if (isa<CXXConstructExpr>(RetE) && RetE->getType()->isRecordType())
    return;

  // The CK_CopyAndAutoreleaseBlockObject cast causes the block to be copied
  // so the stack address is not escaping here.
  if (auto *ICE = dyn_cast<ImplicitCastExpr>(RetE)) {
    if (isa<BlockDataRegion>(R) &&
        ICE->getCastKind() == CK_CopyAndAutoreleaseBlockObject) {
      return;
    }
  }

  EmitStackError(C, R, RetE);
}

void StackAddrEscapeChecker::checkEndFunction(CheckerContext &Ctx) const {
  ProgramStateRef state = Ctx.getState();

  // Iterate over all bindings to global variables and see if it contains
  // a memory region in the stack space.
  class CallBack : public StoreManager::BindingsHandler {
  private:
    CheckerContext &Ctx;
    const StackFrameContext *CurSFC;
  public:
    SmallVector<std::pair<const MemRegion*, const MemRegion*>, 10> V;

    CallBack(CheckerContext &CC) :
      Ctx(CC),
      CurSFC(CC.getLocationContext()->getCurrentStackFrame())
    {}

    bool HandleBinding(StoreManager &SMgr, Store store,
                       const MemRegion *region, SVal val) override {

      if (!isa<GlobalsSpaceRegion>(region->getMemorySpace()))
        return true;

      const MemRegion *vR = val.getAsRegion();
      if (!vR)
        return true;

      // Under automated retain release, it is okay to assign a block
      // directly to a global variable.
      if (Ctx.getASTContext().getLangOpts().ObjCAutoRefCount &&
          isa<BlockDataRegion>(vR))
        return true;

      if (const StackSpaceRegion *SSR =
          dyn_cast<StackSpaceRegion>(vR->getMemorySpace())) {
        // If the global variable holds a location in the current stack frame,
        // record the binding to emit a warning.
        if (SSR->getStackFrame() == CurSFC)
          V.push_back(std::make_pair(region, vR));
      }

      return true;
    }
  };

  CallBack cb(Ctx);
  state->getStateManager().getStoreManager().iterBindings(state->getStore(),cb);

  if (cb.V.empty())
    return;

  // Generate an error node.
  ExplodedNode *N = Ctx.generateNonFatalErrorNode(state);
  if (!N)
    return;

  if (!BT_stackleak)
    BT_stackleak.reset(
        new BuiltinBug(this, "Stack address stored into global variable",
                       "Stack address was saved into a global variable. "
                       "This is dangerous because the address will become "
                       "invalid after returning from the function"));

  for (unsigned i = 0, e = cb.V.size(); i != e; ++i) {
    // Generate a report for this bug.
    SmallString<512> buf;
    llvm::raw_svector_ostream os(buf);
    SourceRange range = genName(os, cb.V[i].second, Ctx.getASTContext());
    os << " is still referred to by the global variable '";
    const VarRegion *VR = cast<VarRegion>(cb.V[i].first->getBaseRegion());
    os << *VR->getDecl()
       << "' upon returning to the caller.  This will be a dangling reference";
    auto report = llvm::make_unique<BugReport>(*BT_stackleak, os.str(), N);
    if (range.isValid())
      report->addRange(range);

    Ctx.emitReport(std::move(report));
  }
}

void ento::registerStackAddrEscapeChecker(CheckerManager &mgr) {
  mgr.registerChecker<StackAddrEscapeChecker>();
}
