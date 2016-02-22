//===- ObjCSuperDeallocChecker.cpp - Check correct use of [super dealloc] -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines ObjCSuperDeallocChecker, a builtin check that warns when
// [super dealloc] is called twice on the same instance in MRR mode.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"

using namespace clang;
using namespace ento;

namespace {
class ObjCSuperDeallocChecker
    : public Checker<check::PostObjCMessage, check::PreObjCMessage> {

  mutable IdentifierInfo *IIdealloc, *IINSObject;
  mutable Selector SELdealloc;

  std::unique_ptr<BugType> DoubleSuperDeallocBugType;

  void initIdentifierInfoAndSelectors(ASTContext &Ctx) const;

  bool isSuperDeallocMessage(const ObjCMethodCall &M) const;

public:
  ObjCSuperDeallocChecker();
  void checkPostObjCMessage(const ObjCMethodCall &M, CheckerContext &C) const;
  void checkPreObjCMessage(const ObjCMethodCall &M, CheckerContext &C) const;
};

} // End anonymous namespace.

// Remember whether [super dealloc] has previously been called on the
// a SymbolRef for the receiver.
REGISTER_SET_WITH_PROGRAMSTATE(CalledSuperDealloc, SymbolRef)

class SuperDeallocBRVisitor final
    : public BugReporterVisitorImpl<SuperDeallocBRVisitor> {

  SymbolRef ReceiverSymbol;
  bool Satisfied;

public:
  SuperDeallocBRVisitor(SymbolRef ReceiverSymbol)
      : ReceiverSymbol(ReceiverSymbol),
        Satisfied(false) {}

  PathDiagnosticPiece *VisitNode(const ExplodedNode *Succ,
                                 const ExplodedNode *Pred,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override;

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.Add(ReceiverSymbol);
  }
};

void ObjCSuperDeallocChecker::checkPreObjCMessage(const ObjCMethodCall &M,
                                                  CheckerContext &C) const {
  if (!isSuperDeallocMessage(M))
    return;

  ProgramStateRef State = C.getState();
  SymbolRef ReceiverSymbol = M.getReceiverSVal().getAsSymbol();
  assert(ReceiverSymbol && "No receiver symbol at call to [super dealloc]?");

  bool AlreadyCalled = State->contains<CalledSuperDealloc>(ReceiverSymbol);

  // If [super dealloc] has not been called, there is nothing to do. We'll
  // note the fact that [super dealloc] was called in checkPostObjCMessage.
  if (!AlreadyCalled)
    return;

  // We have a duplicate [super dealloc] method call.
  // This likely causes a crash, so stop exploring the
  // path by generating a sink.
  ExplodedNode *ErrNode = C.generateErrorNode();
  // If we've already reached this node on another path, return.
  if (!ErrNode)
    return;

  // Generate the report.
  std::unique_ptr<BugReport> BR(
      new BugReport(*DoubleSuperDeallocBugType,
                    "[super dealloc] should not be called multiple times",
                    ErrNode));
  BR->addRange(M.getOriginExpr()->getSourceRange());
  BR->addVisitor(llvm::make_unique<SuperDeallocBRVisitor>(ReceiverSymbol));
  C.emitReport(std::move(BR));

  return;
}

void ObjCSuperDeallocChecker::checkPostObjCMessage(const ObjCMethodCall &M,
                                                   CheckerContext &C) const {
  // Check for [super dealloc] method call.
  if (!isSuperDeallocMessage(M))
    return;

  ProgramStateRef State = C.getState();
  SymbolRef ReceiverSymbol = M.getSelfSVal().getAsSymbol();
  assert(ReceiverSymbol && "No receiver symbol at call to [super dealloc]?");

  // We add this transition in checkPostObjCMessage to avoid warning when
  // we inline a call to [super dealloc] where the inlined call itself
  // calls [super dealloc].
  State = State->add<CalledSuperDealloc>(ReceiverSymbol);
  C.addTransition(State);
}

ObjCSuperDeallocChecker::ObjCSuperDeallocChecker()
    : IIdealloc(nullptr), IINSObject(nullptr) {

  DoubleSuperDeallocBugType.reset(
      new BugType(this, "[super dealloc] should not be called more than once",
                  categories::CoreFoundationObjectiveC));
}

void
ObjCSuperDeallocChecker::initIdentifierInfoAndSelectors(ASTContext &Ctx) const {
  if (IIdealloc)
    return;

  IIdealloc = &Ctx.Idents.get("dealloc");
  IINSObject = &Ctx.Idents.get("NSObject");

  SELdealloc = Ctx.Selectors.getSelector(0, &IIdealloc);
}

bool
ObjCSuperDeallocChecker::isSuperDeallocMessage(const ObjCMethodCall &M) const {
  if (M.getOriginExpr()->getReceiverKind() != ObjCMessageExpr::SuperInstance)
    return false;

  ASTContext &Ctx = M.getState()->getStateManager().getContext();
  initIdentifierInfoAndSelectors(Ctx);

  return M.getSelector() == SELdealloc;
}

PathDiagnosticPiece *SuperDeallocBRVisitor::VisitNode(const ExplodedNode *Succ,
                                                      const ExplodedNode *Pred,
                                                      BugReporterContext &BRC,
                                                      BugReport &BR) {
  if (Satisfied)
    return nullptr;

  ProgramStateRef State = Succ->getState();

  bool CalledNow =
      Succ->getState()->contains<CalledSuperDealloc>(ReceiverSymbol);
  bool CalledBefore =
      Pred->getState()->contains<CalledSuperDealloc>(ReceiverSymbol);

  // Is Succ the node on which the analyzer noted that [super dealloc] was
  // called on ReceiverSymbol?
  if (CalledNow && !CalledBefore) {
    Satisfied = true;

    ProgramPoint P = Succ->getLocation();
    PathDiagnosticLocation L =
        PathDiagnosticLocation::create(P, BRC.getSourceManager());

    if (!L.isValid() || !L.asLocation().isValid())
      return nullptr;

    return new PathDiagnosticEventPiece(
        L, "[super dealloc] called here");
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Checker Registration.
//===----------------------------------------------------------------------===//

void ento::registerObjCSuperDeallocChecker(CheckerManager &Mgr) {
  const LangOptions &LangOpts = Mgr.getLangOpts();
  if (LangOpts.getGC() == LangOptions::GCOnly || LangOpts.ObjCAutoRefCount)
    return;
  Mgr.registerChecker<ObjCSuperDeallocChecker>();
}
