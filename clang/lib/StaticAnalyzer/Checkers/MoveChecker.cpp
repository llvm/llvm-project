// MoveChecker.cpp - Check use of moved-from objects. - C++ ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines checker which checks for potential misuses of a moved-from
// object. That means method calls on the object or copying it in moved-from
// state.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/ExprCXX.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/StringSet.h"

using namespace clang;
using namespace ento;

namespace {

struct RegionState {
private:
  enum Kind { Moved, Reported } K;
  RegionState(Kind InK) : K(InK) {}

public:
  bool isReported() const { return K == Reported; }
  bool isMoved() const { return K == Moved; }

  static RegionState getReported() { return RegionState(Reported); }
  static RegionState getMoved() { return RegionState(Moved); }

  bool operator==(const RegionState &X) const { return K == X.K; }
  void Profile(llvm::FoldingSetNodeID &ID) const { ID.AddInteger(K); }
};

class MoveChecker
    : public Checker<check::PreCall, check::PostCall,
                     check::DeadSymbols, check::RegionChanges> {
public:
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
  void checkPreCall(const CallEvent &MC, CheckerContext &C) const;
  void checkPostCall(const CallEvent &MC, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
  ProgramStateRef
  checkRegionChanges(ProgramStateRef State,
                     const InvalidatedSymbols *Invalidated,
                     ArrayRef<const MemRegion *> RequestedRegions,
                     ArrayRef<const MemRegion *> InvalidatedRegions,
                     const LocationContext *LCtx, const CallEvent *Call) const;
  void printState(raw_ostream &Out, ProgramStateRef State,
                  const char *NL, const char *Sep) const override;

private:
  enum MisuseKind { MK_FunCall, MK_Copy, MK_Move };

  struct ObjectKind {
    bool Local : 1; // Is this a local variable or a local rvalue reference?
    bool STL : 1; // Is this an object of a standard type?
  };

  // Not all of these are entirely move-safe, but they do provide *some*
  // guarantees, and it means that somebody is using them after move
  // in a valid manner.
  // TODO: We can still try to identify *unsafe* use after move, such as
  // dereference of a moved-from smart pointer (which is guaranteed to be null).
  const llvm::StringSet<> StandardMoveSafeClasses = {
      "basic_filebuf",
      "basic_ios",
      "future",
      "optional",
      "packaged_task"
      "promise",
      "shared_future",
      "shared_lock",
      "shared_ptr",
      "thread",
      "unique_ptr",
      "unique_lock",
      "weak_ptr",
  };

  // Obtains ObjectKind of an object. Because class declaration cannot always
  // be easily obtained from the memory region, it is supplied separately.
  ObjectKind classifyObject(const MemRegion *MR, const CXXRecordDecl *RD) const;

  // Classifies the object and dumps a user-friendly description string to
  // the stream. Return value is equivalent to classifyObject.
  ObjectKind explainObject(llvm::raw_ostream &OS,
                           const MemRegion *MR, const CXXRecordDecl *RD) const;

  bool isStandardMoveSafeClass(const CXXRecordDecl *RD) const;

  class MovedBugVisitor : public BugReporterVisitor {
  public:
    MovedBugVisitor(const MoveChecker &Chk,
                    const MemRegion *R, const CXXRecordDecl *RD)
        : Chk(Chk), Region(R), RD(RD), Found(false) {}

    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Region);
      // Don't add RD because it's, in theory, uniquely determined by
      // the region. In practice though, it's not always possible to obtain
      // the declaration directly from the region, that's why we store it
      // in the first place.
    }

    std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) override;

  private:
    const MoveChecker &Chk;
    // The tracked region.
    const MemRegion *Region;
    // The class of the tracked object.
    const CXXRecordDecl *RD;
    bool Found;
  };

  bool IsAggressive = false;

public:
  void setAggressiveness(bool Aggressive) { IsAggressive = Aggressive; }

private:
  mutable std::unique_ptr<BugType> BT;
  ExplodedNode *reportBug(const MemRegion *Region, const CXXRecordDecl *RD,
                          CheckerContext &C, MisuseKind MK) const;
  bool isInMoveSafeContext(const LocationContext *LC) const;
  bool isStateResetMethod(const CXXMethodDecl *MethodDec) const;
  bool isMoveSafeMethod(const CXXMethodDecl *MethodDec) const;
  const ExplodedNode *getMoveLocation(const ExplodedNode *N,
                                      const MemRegion *Region,
                                      CheckerContext &C) const;
};
} // end anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(TrackedRegionMap, const MemRegion *, RegionState)

// If a region is removed all of the subregions needs to be removed too.
static ProgramStateRef removeFromState(ProgramStateRef State,
                                       const MemRegion *Region) {
  if (!Region)
    return State;
  for (auto &E : State->get<TrackedRegionMap>()) {
    if (E.first->isSubRegionOf(Region))
      State = State->remove<TrackedRegionMap>(E.first);
  }
  return State;
}

static bool isAnyBaseRegionReported(ProgramStateRef State,
                                    const MemRegion *Region) {
  for (auto &E : State->get<TrackedRegionMap>()) {
    if (Region->isSubRegionOf(E.first) && E.second.isReported())
      return true;
  }
  return false;
}

static const MemRegion *unwrapRValueReferenceIndirection(const MemRegion *MR) {
  if (const auto *SR = dyn_cast_or_null<SymbolicRegion>(MR)) {
    SymbolRef Sym = SR->getSymbol();
    if (Sym->getType()->isRValueReferenceType())
      if (const MemRegion *OriginMR = Sym->getOriginRegion())
        return OriginMR;
  }
  return MR;
}

std::shared_ptr<PathDiagnosticPiece>
MoveChecker::MovedBugVisitor::VisitNode(const ExplodedNode *N,
                                        BugReporterContext &BRC, BugReport &BR) {
  // We need only the last move of the reported object's region.
  // The visitor walks the ExplodedGraph backwards.
  if (Found)
    return nullptr;
  ProgramStateRef State = N->getState();
  ProgramStateRef StatePrev = N->getFirstPred()->getState();
  const RegionState *TrackedObject = State->get<TrackedRegionMap>(Region);
  const RegionState *TrackedObjectPrev =
      StatePrev->get<TrackedRegionMap>(Region);
  if (!TrackedObject)
    return nullptr;
  if (TrackedObjectPrev && TrackedObject)
    return nullptr;

  // Retrieve the associated statement.
  const Stmt *S = PathDiagnosticLocation::getStmt(N);
  if (!S)
    return nullptr;
  Found = true;

  SmallString<128> Str;
  llvm::raw_svector_ostream OS(Str);

  OS << "Object";
  ObjectKind OK = Chk.explainObject(OS, Region, RD);
  if (OK.STL)
    OS << " is left in a valid but unspecified state after move";
  else
    OS << " is moved";

  // Generate the extra diagnostic.
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, OS.str(), true);
  }

const ExplodedNode *MoveChecker::getMoveLocation(const ExplodedNode *N,
                                                 const MemRegion *Region,
                                                 CheckerContext &C) const {
  // Walk the ExplodedGraph backwards and find the first node that referred to
  // the tracked region.
  const ExplodedNode *MoveNode = N;

  while (N) {
    ProgramStateRef State = N->getState();
    if (!State->get<TrackedRegionMap>(Region))
      break;
    MoveNode = N;
    N = N->pred_empty() ? nullptr : *(N->pred_begin());
  }
  return MoveNode;
}

ExplodedNode *MoveChecker::reportBug(const MemRegion *Region,
                                     const CXXRecordDecl *RD,
                                     CheckerContext &C,
                                     MisuseKind MK) const {
  if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
    if (!BT)
      BT.reset(new BugType(this, "Use-after-move",
                           "C++ move semantics"));

    // Uniqueing report to the same object.
    PathDiagnosticLocation LocUsedForUniqueing;
    const ExplodedNode *MoveNode = getMoveLocation(N, Region, C);

    if (const Stmt *MoveStmt = PathDiagnosticLocation::getStmt(MoveNode))
      LocUsedForUniqueing = PathDiagnosticLocation::createBegin(
          MoveStmt, C.getSourceManager(), MoveNode->getLocationContext());

    // Creating the error message.
    llvm::SmallString<128> Str;
    llvm::raw_svector_ostream OS(Str);
    switch(MK) {
      case MK_FunCall:
        OS << "Method called on moved-from object";
        explainObject(OS, Region, RD);
        break;
      case MK_Copy:
        OS << "Moved-from object";
        explainObject(OS, Region, RD);
        OS << " is copied";
        break;
      case MK_Move:
        OS << "Moved-from object";
        explainObject(OS, Region, RD);
        OS << " is moved";
        break;
    }

    auto R =
        llvm::make_unique<BugReport>(*BT, OS.str(), N, LocUsedForUniqueing,
                                     MoveNode->getLocationContext()->getDecl());
    R->addVisitor(llvm::make_unique<MovedBugVisitor>(*this, Region, RD));
    C.emitReport(std::move(R));
    return N;
  }
  return nullptr;
}

void MoveChecker::checkPostCall(const CallEvent &Call,
                                CheckerContext &C) const {
  const auto *AFC = dyn_cast<AnyFunctionCall>(&Call);
  if (!AFC)
    return;

  ProgramStateRef State = C.getState();
  const auto MethodDecl = dyn_cast_or_null<CXXMethodDecl>(AFC->getDecl());
  if (!MethodDecl)
    return;

  const auto *ConstructorDecl = dyn_cast<CXXConstructorDecl>(MethodDecl);

  const auto *CC = dyn_cast_or_null<CXXConstructorCall>(&Call);
  // Check if an object became moved-from.
  // Object can become moved from after a call to move assignment operator or
  // move constructor .
  if (ConstructorDecl && !ConstructorDecl->isMoveConstructor())
    return;

  if (!ConstructorDecl && !MethodDecl->isMoveAssignmentOperator())
    return;

  const auto ArgRegion = AFC->getArgSVal(0).getAsRegion();
  if (!ArgRegion)
    return;

  // In non-aggressive mode, only warn on use-after-move of local variables (or
  // local rvalue references) and of STL objects. The former is possible because
  // local variables (or local rvalue references) are not tempting their user to
  // re-use the storage. The latter is possible because STL objects are known
  // to end up in a valid but unspecified state after the move and their
  // state-reset methods are also known, which allows us to predict
  // precisely when use-after-move is invalid.
  // In aggressive mode, warn on any use-after-move because the user
  // has intentionally asked us to completely eliminate use-after-move
  // in his code.
  ObjectKind OK = classifyObject(ArgRegion, MethodDecl->getParent());
  if (!IsAggressive && !OK.Local && !OK.STL)
    return;

  // Skip moving the object to itself.
  if (CC && CC->getCXXThisVal().getAsRegion() == ArgRegion)
    return;
  if (const auto *IC = dyn_cast<CXXInstanceCall>(AFC))
    if (IC->getCXXThisVal().getAsRegion() == ArgRegion)
      return;

  const MemRegion *BaseRegion = ArgRegion->getBaseRegion();
  // Skip temp objects because of their short lifetime.
  if (BaseRegion->getAs<CXXTempObjectRegion>() ||
      AFC->getArgExpr(0)->isRValue())
    return;
  // If it has already been reported do not need to modify the state.

  if (State->get<TrackedRegionMap>(ArgRegion))
    return;
  // Mark object as moved-from.
  State = State->set<TrackedRegionMap>(ArgRegion, RegionState::getMoved());
  C.addTransition(State);
}

bool MoveChecker::isMoveSafeMethod(const CXXMethodDecl *MethodDec) const {
  // We abandon the cases where bool/void/void* conversion happens.
  if (const auto *ConversionDec =
          dyn_cast_or_null<CXXConversionDecl>(MethodDec)) {
    const Type *Tp = ConversionDec->getConversionType().getTypePtrOrNull();
    if (!Tp)
      return false;
    if (Tp->isBooleanType() || Tp->isVoidType() || Tp->isVoidPointerType())
      return true;
  }
  // Function call `empty` can be skipped.
  return (MethodDec && MethodDec->getDeclName().isIdentifier() &&
      (MethodDec->getName().lower() == "empty" ||
       MethodDec->getName().lower() == "isempty"));
}

bool MoveChecker::isStateResetMethod(const CXXMethodDecl *MethodDec) const {
  if (!MethodDec)
      return false;
  if (MethodDec->hasAttr<ReinitializesAttr>())
      return true;
  if (MethodDec->getDeclName().isIdentifier()) {
    std::string MethodName = MethodDec->getName().lower();
    // TODO: Some of these methods (eg., resize) are not always resetting
    // the state, so we should consider looking at the arguments.
    if (MethodName == "reset" || MethodName == "clear" ||
        MethodName == "destroy" || MethodName == "resize" ||
        MethodName == "shrink")
      return true;
  }
  return false;
}

// Don't report an error inside a move related operation.
// We assume that the programmer knows what she does.
bool MoveChecker::isInMoveSafeContext(const LocationContext *LC) const {
  do {
    const auto *CtxDec = LC->getDecl();
    auto *CtorDec = dyn_cast_or_null<CXXConstructorDecl>(CtxDec);
    auto *DtorDec = dyn_cast_or_null<CXXDestructorDecl>(CtxDec);
    auto *MethodDec = dyn_cast_or_null<CXXMethodDecl>(CtxDec);
    if (DtorDec || (CtorDec && CtorDec->isCopyOrMoveConstructor()) ||
        (MethodDec && MethodDec->isOverloadedOperator() &&
         MethodDec->getOverloadedOperator() == OO_Equal) ||
        isStateResetMethod(MethodDec) || isMoveSafeMethod(MethodDec))
      return true;
  } while ((LC = LC->getParent()));
  return false;
}

bool MoveChecker::isStandardMoveSafeClass(const CXXRecordDecl *RD) const {
  const IdentifierInfo *II = RD->getIdentifier();
  return II && StandardMoveSafeClasses.count(II->getName());
}

MoveChecker::ObjectKind
MoveChecker::classifyObject(const MemRegion *MR,
                            const CXXRecordDecl *RD) const {
  // Local variables and local rvalue references are classified as "Local".
  // For the purposes of this checker, we classify move-safe STL types
  // as not-"STL" types, because that's how the checker treats them.
  MR = unwrapRValueReferenceIndirection(MR);
  return {
    /*Local=*/
        MR && isa<VarRegion>(MR) && isa<StackSpaceRegion>(MR->getMemorySpace()),
    /*STL=*/
        RD && RD->getDeclContext()->isStdNamespace() &&
        !isStandardMoveSafeClass(RD)
  };
}

MoveChecker::ObjectKind
MoveChecker::explainObject(llvm::raw_ostream &OS, const MemRegion *MR,
                           const CXXRecordDecl *RD) const {
  // We may need a leading space every time we actually explain anything,
  // and we never know if we are to explain anything until we try.
  if (const auto DR =
          dyn_cast_or_null<DeclRegion>(unwrapRValueReferenceIndirection(MR))) {
    const auto *RegionDecl = cast<NamedDecl>(DR->getDecl());
    OS << " '" << RegionDecl->getNameAsString() << "'";
  }
  ObjectKind OK = classifyObject(MR, RD);
  if (OK.STL) {
    OS << " of type '" << RD->getQualifiedNameAsString() << "'";
  }
  return OK;
}

void MoveChecker::checkPreCall(const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const LocationContext *LC = C.getLocationContext();
  ExplodedNode *N = nullptr;

  // Remove the MemRegions from the map on which a ctor/dtor call or assignment
  // happened.

  // Checking constructor calls.
  if (const auto *CC = dyn_cast<CXXConstructorCall>(&Call)) {
    State = removeFromState(State, CC->getCXXThisVal().getAsRegion());
    auto CtorDec = CC->getDecl();
    // Check for copying a moved-from object and report the bug.
    if (CtorDec && CtorDec->isCopyOrMoveConstructor()) {
      const MemRegion *ArgRegion = CC->getArgSVal(0).getAsRegion();
      const RegionState *ArgState = State->get<TrackedRegionMap>(ArgRegion);
      if (ArgState && ArgState->isMoved()) {
        if (!isInMoveSafeContext(LC)) {
          const CXXRecordDecl *RD = CtorDec->getParent();
          if(CtorDec->isMoveConstructor())
            N = reportBug(ArgRegion, RD, C, MK_Move);
          else
            N = reportBug(ArgRegion, RD, C, MK_Copy);
          State = State->set<TrackedRegionMap>(ArgRegion,
                                               RegionState::getReported());
        }
      }
    }
    C.addTransition(State, N);
    return;
  }

  const auto IC = dyn_cast<CXXInstanceCall>(&Call);
  if (!IC)
    return;

  // Calling a destructor on a moved object is fine.
  if (isa<CXXDestructorCall>(IC))
    return;

  const MemRegion *ThisRegion = IC->getCXXThisVal().getAsRegion();
  if (!ThisRegion)
    return;

  const auto MethodDecl = dyn_cast_or_null<CXXMethodDecl>(IC->getDecl());
  if (!MethodDecl)
    return;

  // Store class declaration as well, for bug reporting purposes.
  const CXXRecordDecl *RD = MethodDecl->getParent();

  // Checking assignment operators.
  bool OperatorEq = MethodDecl->isOverloadedOperator() &&
                    MethodDecl->getOverloadedOperator() == OO_Equal;
  // Remove the tracked object for every assignment operator, but report bug
  // only for move or copy assignment's argument.
  if (OperatorEq) {
    State = removeFromState(State, ThisRegion);
    if (MethodDecl->isCopyAssignmentOperator() ||
        MethodDecl->isMoveAssignmentOperator()) {
      const RegionState *ArgState =
          State->get<TrackedRegionMap>(IC->getArgSVal(0).getAsRegion());
      if (ArgState && ArgState->isMoved() && !isInMoveSafeContext(LC)) {
        const MemRegion *ArgRegion = IC->getArgSVal(0).getAsRegion();
        if(MethodDecl->isMoveAssignmentOperator())
          N = reportBug(ArgRegion, RD, C, MK_Move);
        else
          N = reportBug(ArgRegion, RD, C, MK_Copy);
        State =
            State->set<TrackedRegionMap>(ArgRegion, RegionState::getReported());
      }
    }
    C.addTransition(State, N);
    return;
  }

  // The remaining part is check only for method call on a moved-from object.

  // We want to investigate the whole object, not only sub-object of a parent
  // class in which the encountered method defined.
  while (const auto *BR = dyn_cast<CXXBaseObjectRegion>(ThisRegion))
    ThisRegion = BR->getSuperRegion();

  if (isMoveSafeMethod(MethodDecl))
    return;

  if (isStateResetMethod(MethodDecl)) {
    State = removeFromState(State, ThisRegion);
    C.addTransition(State);
    return;
  }

  // If it is already reported then we don't report the bug again.
  const RegionState *ThisState = State->get<TrackedRegionMap>(ThisRegion);
  if (!(ThisState && ThisState->isMoved()))
    return;

  // Don't report it in case if any base region is already reported
  if (isAnyBaseRegionReported(State, ThisRegion))
    return;

  if (isInMoveSafeContext(LC))
    return;

  N = reportBug(ThisRegion, RD, C, MK_FunCall);
  State = State->set<TrackedRegionMap>(ThisRegion, RegionState::getReported());
  C.addTransition(State, N);
}

void MoveChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                   CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  TrackedRegionMapTy TrackedRegions = State->get<TrackedRegionMap>();
  for (TrackedRegionMapTy::value_type E : TrackedRegions) {
    const MemRegion *Region = E.first;
    bool IsRegDead = !SymReaper.isLiveRegion(Region);

    // Remove the dead regions from the region map.
    if (IsRegDead) {
      State = State->remove<TrackedRegionMap>(Region);
    }
  }
  C.addTransition(State);
}

ProgramStateRef MoveChecker::checkRegionChanges(
    ProgramStateRef State, const InvalidatedSymbols *Invalidated,
    ArrayRef<const MemRegion *> RequestedRegions,
    ArrayRef<const MemRegion *> InvalidatedRegions,
    const LocationContext *LCtx, const CallEvent *Call) const {
  if (Call) {
    // Relax invalidation upon function calls: only invalidate parameters
    // that are passed directly via non-const pointers or non-const references
    // or rvalue references.
    // In case of an InstanceCall don't invalidate the this-region since
    // it is fully handled in checkPreCall and checkPostCall.
    const MemRegion *ThisRegion = nullptr;
    if (const auto *IC = dyn_cast<CXXInstanceCall>(Call))
      ThisRegion = IC->getCXXThisVal().getAsRegion();

    // Requested ("explicit") regions are the regions passed into the call
    // directly, but not all of them end up being invalidated.
    // But when they do, they appear in the InvalidatedRegions array as well.
    for (const auto *Region : RequestedRegions) {
      if (ThisRegion != Region) {
        if (llvm::find(InvalidatedRegions, Region) !=
            std::end(InvalidatedRegions)) {
          State = removeFromState(State, Region);
        }
      }
    }
  } else {
    // For invalidations that aren't caused by calls, assume nothing. In
    // particular, direct write into an object's field invalidates the status.
    for (const auto *Region : InvalidatedRegions)
      State = removeFromState(State, Region->getBaseRegion());
  }

  return State;
}

void MoveChecker::printState(raw_ostream &Out, ProgramStateRef State,
                             const char *NL, const char *Sep) const {

  TrackedRegionMapTy RS = State->get<TrackedRegionMap>();

  if (!RS.isEmpty()) {
    Out << Sep << "Moved-from objects :" << NL;
    for (auto I: RS) {
      I.first->dumpToStream(Out);
      if (I.second.isMoved())
        Out << ": moved";
      else
        Out << ": moved and reported";
      Out << NL;
    }
  }
}
void ento::registerMoveChecker(CheckerManager &mgr) {
  MoveChecker *chk = mgr.registerChecker<MoveChecker>();
  chk->setAggressiveness(mgr.getAnalyzerOptions().getCheckerBooleanOption(
      "Aggressive", false, chk));
}
