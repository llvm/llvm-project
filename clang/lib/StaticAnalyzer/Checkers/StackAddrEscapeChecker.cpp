//=== StackAddrEscapeChecker.cpp ----------------------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines stack address leak checker, which checks if an invalid
// stack address is stored into a global or heap location. See CERT DCL30-C.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;
using namespace ento;

namespace {
class StackAddrEscapeChecker
    : public CheckerFamily<check::PreCall, check::PreStmt<ReturnStmt>,
                           check::EndFunction> {
  mutable IdentifierInfo *dispatch_semaphore_tII = nullptr;

public:
  StringRef getDebugTag() const override { return "StackAddrEscapeChecker"; }

  CheckerFrontend StackAddrEscape;
  CheckerFrontend StackAddrAsyncEscape;

  const BugType StackLeak{&StackAddrEscape,
                          "Stack address leaks outside of stack frame"};
  const BugType ReturnStack{&StackAddrEscape,
                            "Return of address to stack-allocated memory"};
  const BugType CapturedStackAsync{
      &StackAddrAsyncEscape, "Address of stack-allocated memory is captured"};

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &Ctx) const;

private:
  void checkAsyncExecutedBlockCaptures(const BlockDataRegion &B,
                                       CheckerContext &C) const;
  void EmitReturnLeakError(CheckerContext &C, const MemRegion *LeakedRegion,
                           const Expr *RetE) const;
  bool isSemaphoreCaptured(const BlockDecl &B) const;
  static SourceRange genName(raw_ostream &os, const MemRegion *R,
                             ASTContext &Ctx);
  static SmallVector<std::pair<const MemRegion *, const StackSpaceRegion *>, 4>
  getCapturedStackRegions(const BlockDataRegion &B, CheckerContext &C);
  static bool isNotInCurrentFrame(const StackSpaceRegion *MS,
                                  CheckerContext &C);
};
} // namespace

SourceRange StackAddrEscapeChecker::genName(raw_ostream &os, const MemRegion *R,
                                            ASTContext &Ctx) {
  // Get the base region, stripping away fields and elements.
  R = R->getBaseRegion();
  SourceManager &SM = Ctx.getSourceManager();
  SourceRange range;
  os << "Address of ";

  // Check if the region is a compound literal.
  if (const auto *CR = dyn_cast<CompoundLiteralRegion>(R)) {
    const CompoundLiteralExpr *CL = CR->getLiteralExpr();
    os << "stack memory associated with a compound literal "
          "declared on line "
       << SM.getExpansionLineNumber(CL->getBeginLoc());
    range = CL->getSourceRange();
  } else if (const auto *AR = dyn_cast<AllocaRegion>(R)) {
    const Expr *ARE = AR->getExpr();
    SourceLocation L = ARE->getBeginLoc();
    range = ARE->getSourceRange();
    os << "stack memory allocated by call to alloca() on line "
       << SM.getExpansionLineNumber(L);
  } else if (const auto *BR = dyn_cast<BlockDataRegion>(R)) {
    const BlockDecl *BD = BR->getCodeRegion()->getDecl();
    SourceLocation L = BD->getBeginLoc();
    range = BD->getSourceRange();
    os << "stack-allocated block declared on line "
       << SM.getExpansionLineNumber(L);
  } else if (const auto *VR = dyn_cast<VarRegion>(R)) {
    os << "stack memory associated with local variable '" << VR->getString()
       << '\'';
    range = VR->getDecl()->getSourceRange();
  } else if (const auto *LER = dyn_cast<CXXLifetimeExtendedObjectRegion>(R)) {
    QualType Ty = LER->getValueType().getLocalUnqualifiedType();
    os << "stack memory associated with temporary object of type '";
    Ty.print(os, Ctx.getPrintingPolicy());
    os << "' lifetime extended by local variable";
    if (const IdentifierInfo *ID = LER->getExtendingDecl()->getIdentifier())
      os << " '" << ID->getName() << '\'';
    range = LER->getExpr()->getSourceRange();
  } else if (const auto *TOR = dyn_cast<CXXTempObjectRegion>(R)) {
    QualType Ty = TOR->getValueType().getLocalUnqualifiedType();
    os << "stack memory associated with temporary object of type '";
    Ty.print(os, Ctx.getPrintingPolicy());
    os << "'";
    range = TOR->getExpr()->getSourceRange();
  } else {
    llvm_unreachable("Invalid region in ReturnStackAddressChecker.");
  }

  return range;
}

bool StackAddrEscapeChecker::isNotInCurrentFrame(const StackSpaceRegion *MS,
                                                 CheckerContext &C) {
  return MS->getStackFrame() != C.getStackFrame();
}

bool StackAddrEscapeChecker::isSemaphoreCaptured(const BlockDecl &B) const {
  if (!dispatch_semaphore_tII)
    dispatch_semaphore_tII = &B.getASTContext().Idents.get("dispatch_semaphore_t");
  for (const auto &C : B.captures()) {
    const auto *T = C.getVariable()->getType()->getAs<TypedefType>();
    if (T && T->getDecl()->getIdentifier() == dispatch_semaphore_tII)
      return true;
  }
  return false;
}

SmallVector<std::pair<const MemRegion *, const StackSpaceRegion *>, 4>
StackAddrEscapeChecker::getCapturedStackRegions(const BlockDataRegion &B,
                                                CheckerContext &C) {
  SmallVector<std::pair<const MemRegion *, const StackSpaceRegion *>, 4>
      Regions;
  ProgramStateRef State = C.getState();
  for (auto Var : B.referenced_vars()) {
    SVal Val = State->getSVal(Var.getCapturedRegion());
    if (const MemRegion *Region = Val.getAsRegion()) {
      if (const auto *Space =
              Region->getMemorySpaceAs<StackSpaceRegion>(State)) {
        Regions.emplace_back(Region, Space);
      }
    }
  }
  return Regions;
}

static void EmitReturnedAsPartOfError(llvm::raw_ostream &OS, SVal ReturnedVal,
                                      const MemRegion *LeakedRegion) {
  if (const MemRegion *ReturnedRegion = ReturnedVal.getAsRegion()) {
    if (isa<BlockDataRegion>(ReturnedRegion)) {
      OS << " is captured by a returned block";
      return;
    }
  }

  // Generic message
  OS << " returned to caller";
}

void StackAddrEscapeChecker::EmitReturnLeakError(CheckerContext &C,
                                                 const MemRegion *R,
                                                 const Expr *RetE) const {
  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;

  // Generate a report for this bug.
  SmallString<128> buf;
  llvm::raw_svector_ostream os(buf);

  // Error message formatting
  SourceRange range = genName(os, R, C.getASTContext());
  EmitReturnedAsPartOfError(os, C.getSVal(RetE), R);

  auto report =
      std::make_unique<PathSensitiveBugReport>(ReturnStack, os.str(), N);
  report->addRange(RetE->getSourceRange());
  if (range.isValid())
    report->addRange(range);
  C.emitReport(std::move(report));
}

void StackAddrEscapeChecker::checkAsyncExecutedBlockCaptures(
    const BlockDataRegion &B, CheckerContext &C) const {
  // There is a not-too-uncommon idiom
  // where a block passed to dispatch_async captures a semaphore
  // and then the thread (which called dispatch_async) is blocked on waiting
  // for the completion of the execution of the block
  // via dispatch_semaphore_wait. To avoid false-positives (for now)
  // we ignore all the blocks which have captured
  // a variable of the type "dispatch_semaphore_t".
  if (isSemaphoreCaptured(*B.getDecl()))
    return;
  auto Regions = getCapturedStackRegions(B, C);
  for (const MemRegion *Region : llvm::make_first_range(Regions)) {
    // The block passed to dispatch_async may capture another block
    // created on the stack. However, there is no leak in this situaton,
    // no matter if ARC or no ARC is enabled:
    // dispatch_async copies the passed "outer" block (via Block_copy)
    // and if the block has captured another "inner" block,
    // the "inner" block will be copied as well.
    if (isa<BlockDataRegion>(Region))
      continue;
    ExplodedNode *N = C.generateNonFatalErrorNode();
    if (!N)
      continue;
    SmallString<128> Buf;
    llvm::raw_svector_ostream Out(Buf);
    SourceRange Range = genName(Out, Region, C.getASTContext());
    Out << " is captured by an asynchronously-executed block";
    auto Report = std::make_unique<PathSensitiveBugReport>(CapturedStackAsync,
                                                           Out.str(), N);
    if (Range.isValid())
      Report->addRange(Range);
    C.emitReport(std::move(Report));
  }
}

void StackAddrEscapeChecker::checkPreCall(const CallEvent &Call,
                                          CheckerContext &C) const {
  if (!StackAddrAsyncEscape.isEnabled())
    return;
  if (!Call.isGlobalCFunction("dispatch_after") &&
      !Call.isGlobalCFunction("dispatch_async"))
    return;
  for (unsigned Idx = 0, NumArgs = Call.getNumArgs(); Idx < NumArgs; ++Idx) {
    if (const BlockDataRegion *B = dyn_cast_or_null<BlockDataRegion>(
            Call.getArgSVal(Idx).getAsRegion()))
      checkAsyncExecutedBlockCaptures(*B, C);
  }
}

/// A visitor made for use with a ScanReachableSymbols scanner, used
/// for finding stack regions within an SVal that live on the current
/// stack frame of the given checker context. This visitor excludes
/// NonParamVarRegion that data is bound to in a BlockDataRegion's
/// bindings, since these are likely uninteresting, e.g., in case a
/// temporary is constructed on the stack, but it captures values
/// that would leak.
class FindStackRegionsSymbolVisitor final : public SymbolVisitor {
  CheckerContext &Ctxt;
  const StackFrameContext *PoppedStackFrame;
  SmallVectorImpl<const MemRegion *> &EscapingStackRegions;

public:
  explicit FindStackRegionsSymbolVisitor(
      CheckerContext &Ctxt,
      SmallVectorImpl<const MemRegion *> &StorageForStackRegions)
      : Ctxt(Ctxt), PoppedStackFrame(Ctxt.getStackFrame()),
        EscapingStackRegions(StorageForStackRegions) {}

  bool VisitSymbol(SymbolRef sym) override { return true; }

  bool VisitMemRegion(const MemRegion *MR) override {
    SaveIfEscapes(MR);

    if (const BlockDataRegion *BDR = MR->getAs<BlockDataRegion>())
      return VisitBlockDataRegionCaptures(BDR);

    return true;
  }

private:
  void SaveIfEscapes(const MemRegion *MR) {
    const auto *SSR = MR->getMemorySpaceAs<StackSpaceRegion>(Ctxt.getState());

    if (!SSR)
      return;

    const StackFrameContext *CapturedSFC = SSR->getStackFrame();
    if (CapturedSFC == PoppedStackFrame ||
        PoppedStackFrame->isParentOf(CapturedSFC))
      EscapingStackRegions.push_back(MR);
  }

  bool VisitBlockDataRegionCaptures(const BlockDataRegion *BDR) {
    for (auto Var : BDR->referenced_vars()) {
      SVal Val = Ctxt.getState()->getSVal(Var.getCapturedRegion());
      const MemRegion *Region = Val.getAsRegion();
      if (Region) {
        SaveIfEscapes(Region);
        VisitMemRegion(Region);
      }
    }

    return false;
  }
};

/// Given some memory regions that are flagged by FindStackRegionsSymbolVisitor,
/// this function filters out memory regions that are being returned that are
/// likely not true leaks:
/// 1. If returning a block data region that has stack memory space
/// 2. If returning a constructed object that has stack memory space
static SmallVector<const MemRegion *> FilterReturnExpressionLeaks(
    const SmallVectorImpl<const MemRegion *> &MaybeEscaped, CheckerContext &C,
    const Expr *RetE, SVal &RetVal) {

  SmallVector<const MemRegion *> WillEscape;

  const MemRegion *RetRegion = RetVal.getAsRegion();

  // Returning a record by value is fine. (In this case, the returned
  // expression will be a copy-constructor, possibly wrapped in an
  // ExprWithCleanups node.)
  if (const ExprWithCleanups *Cleanup = dyn_cast<ExprWithCleanups>(RetE))
    RetE = Cleanup->getSubExpr();
  bool IsConstructExpr =
      isa<CXXConstructExpr>(RetE) && RetE->getType()->isRecordType();

  // The CK_CopyAndAutoreleaseBlockObject cast causes the block to be copied
  // so the stack address is not escaping here.
  bool IsCopyAndAutoreleaseBlockObj = false;
  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(RetE)) {
    IsCopyAndAutoreleaseBlockObj =
        isa_and_nonnull<BlockDataRegion>(RetRegion) &&
        ICE->getCastKind() == CK_CopyAndAutoreleaseBlockObject;
  }

  for (const MemRegion *MR : MaybeEscaped) {
    if (RetRegion == MR && (IsCopyAndAutoreleaseBlockObj || IsConstructExpr))
      continue;

    WillEscape.push_back(MR);
  }

  return WillEscape;
}

/// For use in finding regions that live on the checker context's current
/// stack frame, deep in the SVal representing the return value.
static SmallVector<const MemRegion *>
FindEscapingStackRegions(CheckerContext &C, const Expr *RetE, SVal RetVal) {
  SmallVector<const MemRegion *> FoundStackRegions;

  FindStackRegionsSymbolVisitor Finder(C, FoundStackRegions);
  ScanReachableSymbols Scanner(C.getState(), Finder);
  Scanner.scan(RetVal);

  return FilterReturnExpressionLeaks(FoundStackRegions, C, RetE, RetVal);
}

void StackAddrEscapeChecker::checkPreStmt(const ReturnStmt *RS,
                                          CheckerContext &C) const {
  if (!StackAddrEscape.isEnabled())
    return;

  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
  RetE = RetE->IgnoreParens();

  SVal V = C.getSVal(RetE);

  SmallVector<const MemRegion *> EscapedStackRegions =
      FindEscapingStackRegions(C, RetE, V);

  for (const MemRegion *ER : EscapedStackRegions)
    EmitReturnLeakError(C, ER, RetE);
}

static const MemSpaceRegion *getStackOrGlobalSpaceRegion(ProgramStateRef State,
                                                         const MemRegion *R) {
  assert(R);
  if (const auto *MemSpace = R->getMemorySpace(State);
      isa<StackSpaceRegion, GlobalsSpaceRegion>(MemSpace))
    return MemSpace;

  // If R describes a lambda capture, it will be a symbolic region
  // referring to a field region of another symbolic region.
  if (const auto *SymReg = R->getBaseRegion()->getAs<SymbolicRegion>()) {
    if (const auto *OriginReg = SymReg->getSymbol()->getOriginRegion())
      return getStackOrGlobalSpaceRegion(State, OriginReg);
  }
  return nullptr;
}

static const MemRegion *getOriginBaseRegion(const MemRegion *Reg) {
  Reg = Reg->getBaseRegion();
  while (const auto *SymReg = dyn_cast<SymbolicRegion>(Reg)) {
    const auto *OriginReg = SymReg->getSymbol()->getOriginRegion();
    if (!OriginReg)
      break;
    Reg = OriginReg->getBaseRegion();
  }
  return Reg;
}

static std::optional<std::string> printReferrer(ProgramStateRef State,
                                                const MemRegion *Referrer) {
  assert(Referrer);
  const StringRef ReferrerMemorySpace = [](const MemSpaceRegion *Space) {
    if (isa<StaticGlobalSpaceRegion>(Space))
      return "static";
    if (isa<GlobalsSpaceRegion>(Space))
      return "global";
    assert(isa<StackSpaceRegion>(Space));
    // This case covers top-level and inlined analyses.
    return "caller";
  }(getStackOrGlobalSpaceRegion(State, Referrer));

  while (!Referrer->canPrintPretty()) {
    if (const auto *SymReg = dyn_cast<SymbolicRegion>(Referrer);
        SymReg && SymReg->getSymbol()->getOriginRegion()) {
      Referrer = SymReg->getSymbol()->getOriginRegion()->getBaseRegion();
    } else if (isa<CXXThisRegion>(Referrer)) {
      // Skip members of a class, it is handled in CheckExprLifetime.cpp as
      // warn_bind_ref_member_to_parameter or
      // warn_init_ptr_member_to_parameter_addr
      return std::nullopt;
    } else if (isa<AllocaRegion>(Referrer)) {
      // Skip alloca() regions, they indicate advanced memory management
      // and higher likelihood of CSA false positives.
      return std::nullopt;
    } else {
      assert(false && "Unexpected referrer region type.");
      return std::nullopt;
    }
  }
  assert(Referrer);
  assert(Referrer->canPrintPretty());

  std::string buf;
  llvm::raw_string_ostream os(buf);
  os << ReferrerMemorySpace << " variable ";
  Referrer->printPretty(os);
  return buf;
}

/// Check whether \p Region refers to a freshly minted symbol after an opaque
/// function call.
static bool isInvalidatedSymbolRegion(const MemRegion *Region) {
  const auto *SymReg = Region->getAs<SymbolicRegion>();
  if (!SymReg)
    return false;
  SymbolRef Symbol = SymReg->getSymbol();

  const auto *DerS = dyn_cast<SymbolDerived>(Symbol);
  return DerS && isa_and_nonnull<SymbolConjured>(DerS->getParentSymbol());
}

void StackAddrEscapeChecker::checkEndFunction(const ReturnStmt *RS,
                                              CheckerContext &Ctx) const {
  if (!StackAddrEscape.isEnabled())
    return;

  ExplodedNode *Node = Ctx.getPredecessor();

  bool ExitingTopFrame =
      Ctx.getPredecessor()->getLocationContext()->inTopFrame();

  if (ExitingTopFrame &&
      Node->getLocation().getTag() == ExprEngine::cleanupNodeTag() &&
      Node->getFirstPred()) {
    // When finishing analysis of a top-level function, engine proactively
    // removes dead symbols thus preventing this checker from looking through
    // the output parameters. Take 1 step back, to the node where these symbols
    // and their bindings are still present
    Node = Node->getFirstPred();
  }

  // Iterate over all bindings to global variables and see if it contains
  // a memory region in the stack space.
  class CallBack : public StoreManager::BindingsHandler {
  private:
    CheckerContext &Ctx;
    ProgramStateRef State;
    const StackFrameContext *PoppedFrame;
    const bool TopFrame;

    /// Look for stack variables referring to popped stack variables.
    /// Returns true only if it found some dangling stack variables
    /// referred by an other stack variable from different stack frame.
    bool checkForDanglingStackVariable(const MemRegion *Referrer,
                                       const MemRegion *Referred) {
      const auto *ReferrerMemSpace =
          getStackOrGlobalSpaceRegion(State, Referrer);
      const auto *ReferredMemSpace =
          Referred->getMemorySpaceAs<StackSpaceRegion>(State);

      if (!ReferrerMemSpace || !ReferredMemSpace)
        return false;

      const auto *ReferrerStackSpace =
          ReferrerMemSpace->getAs<StackSpaceRegion>();

      if (!ReferrerStackSpace)
        return false;

      if (const auto *ReferredFrame = ReferredMemSpace->getStackFrame();
          ReferredFrame != PoppedFrame) {
        return false;
      }

      if (ReferrerStackSpace->getStackFrame()->isParentOf(PoppedFrame)) {
        V.emplace_back(Referrer, Referred);
        return true;
      }
      if (isa<StackArgumentsSpaceRegion>(ReferrerMemSpace) &&
          // Not a simple ptr (int*) but something deeper, e.g. int**
          isa<SymbolicRegion>(Referrer->getBaseRegion()) &&
          ReferrerStackSpace->getStackFrame() == PoppedFrame && TopFrame) {
        // Output parameter of a top-level function
        V.emplace_back(Referrer, Referred);
        return true;
      }
      return false;
    }

    // Keep track of the variables that were invalidated through an opaque
    // function call. Even if the initial values of such variables were bound to
    // an address of a local variable, we cannot claim anything now, at the
    // function exit, so skip them to avoid false positives.
    void recordInInvalidatedRegions(const MemRegion *Region) {
      if (isInvalidatedSymbolRegion(Region))
        ExcludedRegions.insert(getOriginBaseRegion(Region));
    }

  public:
    SmallVector<std::pair<const MemRegion *, const MemRegion *>, 10> V;
    // ExcludedRegions are skipped from reporting.
    // I.e., if a referrer in this set, skip the related bug report.
    // It is useful to avoid false positive for the variables that were
    // reset to a conjured value after an opaque function call.
    llvm::SmallPtrSet<const MemRegion *, 4> ExcludedRegions;

    CallBack(CheckerContext &CC, bool TopFrame)
        : Ctx(CC), State(CC.getState()), PoppedFrame(CC.getStackFrame()),
          TopFrame(TopFrame) {}

    bool HandleBinding(StoreManager &SMgr, Store S, const MemRegion *Region,
                       SVal Val) override {
      recordInInvalidatedRegions(Region);
      const MemRegion *VR = Val.getAsRegion();
      if (!VR)
        return true;

      if (checkForDanglingStackVariable(Region, VR))
        return true;

      // Check the globals for the same.
      if (!isa_and_nonnull<GlobalsSpaceRegion>(
              getStackOrGlobalSpaceRegion(State, Region)))
        return true;

      if (VR) {
        if (const auto *S = VR->getMemorySpaceAs<StackSpaceRegion>(State);
            S && !isNotInCurrentFrame(S, Ctx)) {
          V.emplace_back(Region, VR);
        }
      }
      return true;
    }
  };

  CallBack Cb(Ctx, ExitingTopFrame);
  ProgramStateRef State = Node->getState();
  State->getStateManager().getStoreManager().iterBindings(State->getStore(),
                                                          Cb);

  if (Cb.V.empty())
    return;

  // Generate an error node.
  ExplodedNode *N = Ctx.generateNonFatalErrorNode(State, Node);
  if (!N)
    return;

  for (const auto &P : Cb.V) {
    const MemRegion *Referrer = P.first->getBaseRegion();
    const MemRegion *Referred = P.second;
    if (Cb.ExcludedRegions.contains(getOriginBaseRegion(Referrer))) {
      continue;
    }

    // Generate a report for this bug.
    const StringRef CommonSuffix =
        " upon returning to the caller.  This will be a dangling reference";
    SmallString<128> Buf;
    llvm::raw_svector_ostream Out(Buf);
    const SourceRange Range = genName(Out, Referred, Ctx.getASTContext());

    if (isa<CXXTempObjectRegion, CXXLifetimeExtendedObjectRegion>(Referrer)) {
      Out << " is still referred to by a temporary object on the stack"
          << CommonSuffix;
      auto Report =
          std::make_unique<PathSensitiveBugReport>(StackLeak, Out.str(), N);
      if (Range.isValid())
        Report->addRange(Range);
      Ctx.emitReport(std::move(Report));
      return;
    }

    auto ReferrerVariable = printReferrer(State, Referrer);
    if (!ReferrerVariable) {
      continue;
    }

    Out << " is still referred to by the " << *ReferrerVariable << CommonSuffix;
    auto Report =
        std::make_unique<PathSensitiveBugReport>(StackLeak, Out.str(), N);
    if (Range.isValid())
      Report->addRange(Range);

    Ctx.emitReport(std::move(Report));
  }
}

#define REGISTER_CHECKER(NAME)                                                 \
  void ento::register##NAME##Checker(CheckerManager &Mgr) {                    \
    Mgr.getChecker<StackAddrEscapeChecker>()->NAME.enable(Mgr);                \
  }                                                                            \
                                                                               \
  bool ento::shouldRegister##NAME##Checker(const CheckerManager &) {           \
    return true;                                                               \
  }

REGISTER_CHECKER(StackAddrEscape)
REGISTER_CHECKER(StackAddrAsyncEscape)
