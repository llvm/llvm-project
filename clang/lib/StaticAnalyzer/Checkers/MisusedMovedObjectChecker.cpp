// MisusedMovedObjectChecker.cpp - Check use of moved-from objects. - C++ -===//
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

class MisusedMovedObjectChecker
    : public Checker<check::PreCall, check::PostCall, check::EndFunction,
                     check::DeadSymbols, check::RegionChanges> {
public:
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
  void checkPreCall(const CallEvent &MC, CheckerContext &C) const;
  void checkPostCall(const CallEvent &MC, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
  ProgramStateRef
  checkRegionChanges(ProgramStateRef State,
                     const InvalidatedSymbols *Invalidated,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const LocationContext *LCtx, const CallEvent *Call) const;
  void printState(raw_ostream &Out, ProgramStateRef State,
                  const char *NL, const char *Sep) const override;

private:
  enum MisuseKind {MK_FunCall, MK_Copy, MK_Move};
  class MovedBugVisitor : public BugReporterVisitor {
  public:
    MovedBugVisitor(const MemRegion *R) : Region(R), Found(false) {}

    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Region);
    }

    std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
                                                   const ExplodedNode *PrevN,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) override;

  private:
    // The tracked region.
    const MemRegion *Region;
    bool Found;
  };

  mutable std::unique_ptr<BugType> BT;
  ExplodedNode *reportBug(const MemRegion *Region, const CallEvent &Call,
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

std::shared_ptr<PathDiagnosticPiece>
MisusedMovedObjectChecker::MovedBugVisitor::VisitNode(const ExplodedNode *N,
                                                      const ExplodedNode *PrevN,
                                                      BugReporterContext &BRC,
                                                      BugReport &BR) {
  // We need only the last move of the reported object's region.
  // The visitor walks the ExplodedGraph backwards.
  if (Found)
    return nullptr;
  ProgramStateRef State = N->getState();
  ProgramStateRef StatePrev = PrevN->getState();
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

  std::string ObjectName;
  if (const auto DecReg = Region->getAs<DeclRegion>()) {
    const auto *RegionDecl = dyn_cast<NamedDecl>(DecReg->getDecl());
    ObjectName = RegionDecl->getNameAsString();
  }
  std::string InfoText;
  if (ObjectName != "")
    InfoText = "'" + ObjectName + "' became 'moved-from' here";
  else
    InfoText = "Became 'moved-from' here";

  // Generate the extra diagnostic.
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, InfoText, true);
}

const ExplodedNode *MisusedMovedObjectChecker::getMoveLocation(
    const ExplodedNode *N, const MemRegion *Region, CheckerContext &C) const {
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

ExplodedNode *MisusedMovedObjectChecker::reportBug(const MemRegion *Region,
                                                   const CallEvent &Call,
                                                   CheckerContext &C,
                                                   MisuseKind MK) const {
  if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
    if (!BT)
      BT.reset(new BugType(this, "Usage of a 'moved-from' object",
                           "C++ move semantics"));

    // Uniqueing report to the same object.
    PathDiagnosticLocation LocUsedForUniqueing;
    const ExplodedNode *MoveNode = getMoveLocation(N, Region, C);

    if (const Stmt *MoveStmt = PathDiagnosticLocation::getStmt(MoveNode))
      LocUsedForUniqueing = PathDiagnosticLocation::createBegin(
          MoveStmt, C.getSourceManager(), MoveNode->getLocationContext());

    // Creating the error message.
    std::string ErrorMessage;
    switch(MK) {
      case MK_FunCall:
        ErrorMessage = "Method call on a 'moved-from' object";
        break;
      case MK_Copy:
        ErrorMessage = "Copying a 'moved-from' object";
        break;
      case MK_Move:
        ErrorMessage = "Moving a 'moved-from' object";
        break;
    }
    if (const auto DecReg = Region->getAs<DeclRegion>()) {
      const auto *RegionDecl = dyn_cast<NamedDecl>(DecReg->getDecl());
      ErrorMessage += " '" + RegionDecl->getNameAsString() + "'";
    }

    auto R =
        llvm::make_unique<BugReport>(*BT, ErrorMessage, N, LocUsedForUniqueing,
                                     MoveNode->getLocationContext()->getDecl());
    R->addVisitor(llvm::make_unique<MovedBugVisitor>(Region));
    C.emitReport(std::move(R));
    return N;
  }
  return nullptr;
}

// Removing the function parameters' MemRegion from the state. This is needed
// for PODs where the trivial destructor does not even created nor executed.
void MisusedMovedObjectChecker::checkEndFunction(const ReturnStmt *RS,
                                                 CheckerContext &C) const {
  auto State = C.getState();
  TrackedRegionMapTy Objects = State->get<TrackedRegionMap>();
  if (Objects.isEmpty())
    return;

  auto LC = C.getLocationContext();

  const auto LD = dyn_cast_or_null<FunctionDecl>(LC->getDecl());
  if (!LD)
    return;
  llvm::SmallSet<const MemRegion *, 8> InvalidRegions;

  for (auto Param : LD->parameters()) {
    auto Type = Param->getType().getTypePtrOrNull();
    if (!Type)
      continue;
    if (!Type->isPointerType() && !Type->isReferenceType()) {
      InvalidRegions.insert(State->getLValue(Param, LC).getAsRegion());
    }
  }

  if (InvalidRegions.empty())
    return;

  for (const auto &E : State->get<TrackedRegionMap>()) {
    if (InvalidRegions.count(E.first->getBaseRegion()))
      State = State->remove<TrackedRegionMap>(E.first);
  }

  C.addTransition(State);
}

void MisusedMovedObjectChecker::checkPostCall(const CallEvent &Call,
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

bool MisusedMovedObjectChecker::isMoveSafeMethod(
    const CXXMethodDecl *MethodDec) const {
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
  if (MethodDec && MethodDec->getDeclName().isIdentifier() &&
      (MethodDec->getName().lower() == "empty" ||
       MethodDec->getName().lower() == "isempty"))
    return true;

  return false;
}

bool MisusedMovedObjectChecker::isStateResetMethod(
    const CXXMethodDecl *MethodDec) const {
  if (MethodDec && MethodDec->getDeclName().isIdentifier()) {
    std::string MethodName = MethodDec->getName().lower();
    if (MethodName == "reset" || MethodName == "clear" ||
        MethodName == "destroy")
      return true;
  }
  return false;
}

// Don't report an error inside a move related operation.
// We assume that the programmer knows what she does.
bool MisusedMovedObjectChecker::isInMoveSafeContext(
    const LocationContext *LC) const {
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

void MisusedMovedObjectChecker::checkPreCall(const CallEvent &Call,
                                             CheckerContext &C) const {
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
          if(CtorDec->isMoveConstructor())
            N = reportBug(ArgRegion, Call, C, MK_Move);
          else
            N = reportBug(ArgRegion, Call, C, MK_Copy);
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
  // In case of destructor call we do not track the object anymore.
  const MemRegion *ThisRegion = IC->getCXXThisVal().getAsRegion();
  if (!ThisRegion)
    return;

  if (dyn_cast_or_null<CXXDestructorDecl>(Call.getDecl())) {
    State = removeFromState(State, ThisRegion);
    C.addTransition(State);
    return;
  }

  const auto MethodDecl = dyn_cast_or_null<CXXMethodDecl>(IC->getDecl());
  if (!MethodDecl)
    return;
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
          N = reportBug(ArgRegion, Call, C, MK_Move);
        else
          N = reportBug(ArgRegion, Call, C, MK_Copy);
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
  while (const CXXBaseObjectRegion *BR =
             dyn_cast<CXXBaseObjectRegion>(ThisRegion))
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

  N = reportBug(ThisRegion, Call, C, MK_FunCall);
  State = State->set<TrackedRegionMap>(ThisRegion, RegionState::getReported());
  C.addTransition(State, N);
}

void MisusedMovedObjectChecker::checkDeadSymbols(SymbolReaper &SymReaper,
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

ProgramStateRef MisusedMovedObjectChecker::checkRegionChanges(
    ProgramStateRef State, const InvalidatedSymbols *Invalidated,
    ArrayRef<const MemRegion *> ExplicitRegions,
    ArrayRef<const MemRegion *> Regions, const LocationContext *LCtx,
    const CallEvent *Call) const {
  // In case of an InstanceCall don't remove the ThisRegion from the GDM since
  // it is handled in checkPreCall and checkPostCall.
  const MemRegion *ThisRegion = nullptr;
  if (const auto *IC = dyn_cast_or_null<CXXInstanceCall>(Call)) {
    ThisRegion = IC->getCXXThisVal().getAsRegion();
  }

  for (ArrayRef<const MemRegion *>::iterator I = ExplicitRegions.begin(),
                                             E = ExplicitRegions.end();
       I != E; ++I) {
    const auto *Region = *I;
    if (ThisRegion != Region) {
      State = removeFromState(State, Region);
    }
  }

  return State;
}

void MisusedMovedObjectChecker::printState(raw_ostream &Out,
                                           ProgramStateRef State,
                                           const char *NL,
                                           const char *Sep) const {

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
void ento::registerMisusedMovedObjectChecker(CheckerManager &mgr) {
  mgr.registerChecker<MisusedMovedObjectChecker>();
}
