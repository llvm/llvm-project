//===- PlacementSelfCopyChecker.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker that finds placement new expressions whose initializer
// reads a part of the target storage that has already been overwritten, e.g.
// `::new (p) T(*p)`. A copy or move constructor that is not inlined is
// reported whenever its source is the target storage.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMap.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

using PlacementKey = std::pair<const CXXNewExpr *, const StackFrame *>;
REGISTER_MAP_WITH_PROGRAMSTATE(PlacementTargets, PlacementKey,
                               const MemRegion *)
REGISTER_SET_WITH_PROGRAMSTATE(SelfSourcedTargets, const MemRegion *)
REGISTER_SET_WITH_PROGRAMSTATE(OverwrittenParts, const MemRegion *)
REGISTER_TRAIT_WITH_PROGRAMSTATE(PendingSelfCopy, const MemRegion *)
REGISTER_TRAIT_WITH_PROGRAMSTATE(ReadOverwritten, const MemRegion *)

namespace {
class PlacementSelfCopyChecker
    : public Checker<check::PostCall, check::PreCall, check::Bind,
                     check::Location, check::PostStmt<CXXNewExpr>> {
  const BugType BT{this, "Placement new initialized from its own storage",
                   categories::MemoryError};
  const BugType ReadBT{this, "Placement new reads overwritten storage",
                       categories::MemoryError};

  void reportRead(const Expr *E, const MemRegion *Overwritten,
                  CheckerContext &C) const;
  void reportSelfCopy(const Expr *Arg, CheckerContext &C) const;
  void reportPendingSelfCopy(const CallEvent &Call, CheckerContext &C) const;
  void checkOverwrittenRead(const Stmt *S, const Expr *E,
                            const MemRegion *Source, CheckerContext &C) const;

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkBind(SVal Loc, SVal Val, const Stmt *S, bool AtDeclInit,
                 CheckerContext &C) const;
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const;
  void checkPostStmt(const CXXNewExpr *NE, CheckerContext &C) const;
};
} // namespace

namespace {
struct Extent {
  const MemRegion *Base;
  int64_t Begin;
  std::optional<int64_t> End;
};
} // namespace

static std::optional<Extent> getExtent(const MemRegion *R, ASTContext &Ctx) {
  RegionOffset Offset = R->getAsOffset();
  if (!Offset.isValid() || Offset.hasSymbolicOffset())
    return std::nullopt;
  Extent E{Offset.getRegion(), Offset.getOffset(), std::nullopt};
  if (const auto *TR = dyn_cast<TypedValueRegion>(R)) {
    QualType Ty = TR->getValueType();
    if (!Ty->isIncompleteType())
      E.End = E.Begin + Ctx.getTypeSize(Ty);
  }
  return E;
}

static bool overlaps(const MemRegion *A, const MemRegion *B, ASTContext &Ctx) {
  if (A == B || A->isSubRegionOf(B) || B->isSubRegionOf(A))
    return true;
  std::optional<Extent> EA = getExtent(A, Ctx), EB = getExtent(B, Ctx);
  if (!EA || !EB || EA->Base != EB->Base)
    return false;
  return (!EB->End || EA->Begin < *EB->End) &&
         (!EA->End || EB->Begin < *EA->End);
}

static bool isWithin(const MemRegion *R, const MemRegion *Container,
                     ASTContext &Ctx) {
  if (R == Container || R->isSubRegionOf(Container))
    return true;
  std::optional<Extent> ER = getExtent(R, Ctx), EC = getExtent(Container, Ctx);
  if (!ER || !EC || ER->Base != EC->Base || ER->Begin < EC->Begin)
    return false;
  return !EC->End || (ER->End && *ER->End <= *EC->End);
}

static bool isThisBased(const Expr *E) {
  while (E) {
    E = E->IgnoreParenCasts();
    if (isa<CXXThisExpr>(E))
      return true;
    if (const auto *ME = dyn_cast<MemberExpr>(E))
      E = ME->getBase();
    else if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E))
      E = ASE->getBase();
    else if (const auto *BO = dyn_cast<BinaryOperator>(E);
             BO && BO->isAssignmentOp())
      E = BO->getLHS();
    else if (const auto *UO = dyn_cast<UnaryOperator>(E);
             UO && UO->getOpcode() == UO_Deref)
      E = UO->getSubExpr();
    else
      return false;
  }
  return false;
}

static const MemRegion *findSelfSourcedTarget(ProgramStateRef State,
                                              const MemRegion *R,
                                              CheckerContext &C) {
  for (const MemRegion *Target : State->get<SelfSourcedTargets>())
    if (isWithin(R, Target, C.getASTContext()))
      return Target;
  return nullptr;
}

static const MemRegion *findOverwritten(ProgramStateRef State,
                                        const MemRegion *R, CheckerContext &C) {
  for (const MemRegion *W : State->get<OverwrittenParts>())
    if (overlaps(R, W, C.getASTContext()))
      return W;
  return nullptr;
}

// CSA binds an init list only after evaluating all elements; recompute order.
static const MemRegion *walkInitList(PlacementKey Key, const MemRegion *Target,
                                     const Stmt *S, const MemRegion *Source,
                                     const MemRegion *&Before,
                                     CheckerContext &C) {
  Before = nullptr;
  const auto &[NE, Frame] = Key;
  if (Frame != C.getStackFrame())
    return nullptr;
  const auto *Init = dyn_cast_if_present<InitListExpr>(NE->getInitializer());
  if (!Init)
    return nullptr;

  const ParentMap &PM = C.getCurrentAnalysisDeclContext()->getParentMap();
  SmallVector<std::pair<const InitListExpr *, unsigned>, 4> Path;
  for (const Stmt *Child = S, *Parent = PM.getParent(S); Parent;
       Child = Parent, Parent = PM.getParent(Parent)) {
    if (const auto *ILE = dyn_cast<InitListExpr>(Parent)) {
      ArrayRef<Expr *> Inits = ILE->inits();
      const auto *It = llvm::find(Inits, Child);
      if (It == Inits.end())
        return nullptr;
      Path.emplace_back(ILE, It - Inits.begin());
    }
    if (Parent == Init)
      break;
  }
  if (Path.empty() || Path.back().first != Init)
    return nullptr;

  ASTContext &Ctx = C.getASTContext();
  MemRegionManager &MRM = C.getStoreManager().getRegionManager();
  SValBuilder &SVB = C.getSValBuilder();
  const auto *R = cast<SubRegion>(Target);
  for (const auto &[ILE, Index] : llvm::reverse(Path)) {
    QualType Ty = ILE->getType();
    if (const auto *AT = Ctx.getAsArrayType(Ty)) {
      QualType ElemTy = AT->getElementType();
      for (unsigned I = 0; I != Index && Source && !Before; ++I) {
        const MemRegion *Elem =
            MRM.getElementRegion(ElemTy, SVB.makeArrayIndex(I), R, Ctx);
        if (overlaps(Source, Elem, Ctx))
          Before = Elem;
      }
      R = MRM.getElementRegion(ElemTy, SVB.makeArrayIndex(Index), R, Ctx);
      continue;
    }

    const auto *RD = Ty->getAsCXXRecordDecl();
    if (!RD || RD->isUnion() || RD->getNumBases())
      return nullptr;
    unsigned I = 0;
    const FieldRegion *Next = nullptr;
    for (const FieldDecl *FD : RD->fields()) {
      if (FD->isUnnamedBitField())
        continue;
      const FieldRegion *Field = MRM.getFieldRegion(FD, R);
      if (I++ == Index) {
        Next = Field;
        break;
      }
      if (Source && !Before && overlaps(Source, Field, Ctx))
        Before = Field;
    }
    if (!Next)
      return nullptr;
    R = Next;
  }
  return R;
}

static const MemRegion *getDestination(PlacementKey Key,
                                       const MemRegion *Target,
                                       const CXXConstructorCall &Ctor,
                                       const MemRegion *This,
                                       CheckerContext &C) {
  const Stmt *S = Ctor.getOriginExpr();
  if (Key.first->getConstructExpr() == S)
    return Key.second == C.getStackFrame() ? This : nullptr;
  const MemRegion *Before;
  return walkInitList(Key, Target, S, nullptr, Before, C);
}

void PlacementSelfCopyChecker::checkOverwrittenRead(const Stmt *S,
                                                    const Expr *E,
                                                    const MemRegion *Source,
                                                    CheckerContext &C) const {
  if (isThisBased(E))
    return;
  ProgramStateRef State = C.getState();
  Source = Source->StripCasts();
  const MemRegion *Overwritten = findSelfSourcedTarget(State, Source, C)
                                     ? findOverwritten(State, Source, C)
                                     : nullptr;
  for (const auto &[Key, Target] : State->get<PlacementTargets>()) {
    if (Overwritten)
      break;
    walkInitList(Key, Target, S, Source, Overwritten, C);
  }
  if (Overwritten)
    reportRead(E, Overwritten, C);
}

void PlacementSelfCopyChecker::reportRead(const Expr *E,
                                          const MemRegion *Overwritten,
                                          CheckerContext &C) const {
  ExplodedNode *N =
      C.generateErrorNode(C.getState()->set<ReadOverwritten>(Overwritten));
  if (!N)
    return;
  auto R = std::make_unique<PathSensitiveBugReport>(
      ReadBT,
      "Placement new reads a part of its storage that it has already "
      "overwritten",
      N);
  R->addRange(E->getSourceRange());
  C.emitReport(std::move(R));
}

void PlacementSelfCopyChecker::reportSelfCopy(const Expr *Arg,
                                              CheckerContext &C) const {
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto R = std::make_unique<PathSensitiveBugReport>(
      BT,
      "Object constructed by placement new is initialized from the storage "
      "it is placed into",
      N);
  R->addRange(Arg->getSourceRange());
  bugreporter::trackExpressionValue(N, bugreporter::getDerefExpr(Arg), *R);
  C.emitReport(std::move(R));
}

// Reports inside the standard library are suppressed; report at the caller.
void PlacementSelfCopyChecker::reportPendingSelfCopy(const CallEvent &Call,
                                                     CheckerContext &C) const {
  const MemRegion *Target = C.getState()->get<PendingSelfCopy>();
  const Expr *Origin = dyn_cast_if_present<Expr>(Call.getOriginExpr());
  if (!Target || !Origin ||
      C.getSourceManager().isInSystemHeader(Origin->getBeginLoc()))
    return;
  for (unsigned I = 0, E = Call.getNumArgs(); I != E; ++I) {
    const MemRegion *Arg = Call.getArgSVal(I).getAsRegion();
    if (Arg && !Call.getArgExpr(I)->getType()->isPointerType() &&
        overlaps(Arg->StripCasts(), Target, C.getASTContext())) {
      reportSelfCopy(Call.getArgExpr(I), C);
      return;
    }
  }
  C.addTransition(C.getState()->remove<PendingSelfCopy>());
}

void PlacementSelfCopyChecker::checkPostCall(const CallEvent &Call,
                                             CheckerContext &C) const {
  if (const auto *Alloc = dyn_cast<CXXAllocatorCall>(&Call)) {
    const CXXNewExpr *NE = Alloc->getOriginExpr();
    const FunctionDecl *OperatorNew = NE->getOperatorNew();
    if (!OperatorNew || !OperatorNew->isReservedGlobalPlacementOperator())
      return;
    const MemRegion *Target = Call.getReturnValue().getAsRegion();
    if (!Target)
      return;
    C.addTransition(C.getState()->set<PlacementTargets>({NE, C.getStackFrame()},
                                                        Target->StripCasts()));
    return;
  }

  const auto *Ctor = dyn_cast<CXXConstructorCall>(&Call);
  if (!Ctor) {
    reportPendingSelfCopy(Call, C);
    return;
  }
  const auto *CD = dyn_cast_if_present<CXXConstructorDecl>(Ctor->getDecl());
  if (!CD || !CD->isCopyOrMoveConstructor())
    return;
  const MemRegion *This = Ctor->getCXXThisVal().getAsRegion();
  const MemRegion *Source = Ctor->getArgSVal(0).getAsRegion();
  if (!This || !Source)
    return;
  This = This->StripCasts();
  Source = Source->StripCasts();

  ProgramStateRef State = C.getState();
  const Stmt *S = Ctor->getOriginExpr();
  for (const auto &[Key, Target] : State->get<PlacementTargets>()) {
    const MemRegion *Dest = getDestination(Key, Target, *Ctor, This, C);
    if (!Dest)
      continue;
    // Inlined constructors are checked per read unless built in a temporary.
    if (C.wasInlined && isWithin(This, Target, C.getASTContext()))
      return;
    if (!overlaps(Source, Dest, C.getASTContext()))
      return;
    if (C.getSourceManager().isInSystemHeader(S->getBeginLoc()))
      C.addTransition(State->set<PendingSelfCopy>(Target));
    else
      reportSelfCopy(Ctor->getArgExpr(0), C);
    return;
  }
}

void PlacementSelfCopyChecker::checkPreCall(const CallEvent &Call,
                                            CheckerContext &C) const {
  const auto *Ctor = dyn_cast<CXXConstructorCall>(&Call);
  if (!Ctor)
    return;
  ProgramStateRef State = C.getState();
  if (State->get<PlacementTargets>().isEmpty())
    return;
  ArrayRef<ParmVarDecl *> Params = Ctor->parameters();
  unsigned NumArgs = std::min<size_t>(Ctor->getNumArgs(), Params.size());
  for (unsigned I = 0; I != NumArgs; ++I) {
    if (!Params[I]->getType()->isReferenceType())
      continue;
    if (const MemRegion *Source = Ctor->getArgSVal(I).getAsRegion()) {
      checkOverwrittenRead(Ctor->getOriginExpr(), Ctor->getArgExpr(I), Source,
                           C);
      if (C.isDifferent())
        return;
    }
  }

  const MemRegion *This = Ctor->getCXXThisVal().getAsRegion();
  if (!This)
    return;
  This = This->StripCasts();
  for (const auto &[Key, Target] : State->get<PlacementTargets>()) {
    const MemRegion *Dest = getDestination(Key, Target, *Ctor, This, C);
    if (!Dest)
      continue;
    for (unsigned I = 0; I != NumArgs; ++I) {
      const MemRegion *Source = Ctor->getArgSVal(I).getAsRegion();
      if (Params[I]->getType()->isReferenceType() && Source &&
          overlaps(Source->StripCasts(), Dest, C.getASTContext())) {
        C.addTransition(State->add<SelfSourcedTargets>(Target));
        return;
      }
    }
    return;
  }
}

void PlacementSelfCopyChecker::checkBind(SVal Loc, SVal Val, const Stmt *S,
                                         bool AtDeclInit,
                                         CheckerContext &C) const {
  const MemRegion *R = Loc.getAsRegion();
  if (!R)
    return;
  R = R->StripCasts();
  ProgramStateRef State = C.getState();
  if (!findSelfSourcedTarget(State, R, C) ||
      State->contains<OverwrittenParts>(R))
    return;
  const NoteTag *Note =
      C.getNoteTag([this, R](PathSensitiveBugReport &BR) -> std::string {
        if (&BR.getBugType() != &ReadBT ||
            BR.getErrorNode()->getState()->get<ReadOverwritten>() != R)
          return "";
        return "Part of the storage is overwritten here";
      });
  C.addTransition(State->add<OverwrittenParts>(R), Note);
}

void PlacementSelfCopyChecker::checkLocation(SVal Loc, bool IsLoad,
                                             const Stmt *S,
                                             CheckerContext &C) const {
  if (!IsLoad || C.getState()->get<PlacementTargets>().isEmpty())
    return;
  const auto *E = dyn_cast_if_present<Expr>(S);
  const MemRegion *R = Loc.getAsRegion();
  if (E && R)
    checkOverwrittenRead(S, E, R, C);
}

void PlacementSelfCopyChecker::checkPostStmt(const CXXNewExpr *NE,
                                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  PlacementKey Key{NE, C.getStackFrame()};
  const MemRegion *const *Target = State->get<PlacementTargets>(Key);
  if (!Target)
    return;
  State = State->remove<PlacementTargets>(Key);
  State = State->remove<SelfSourcedTargets>(*Target);
  for (const MemRegion *W : State->get<OverwrittenParts>())
    if (isWithin(W, *Target, C.getASTContext()))
      State = State->remove<OverwrittenParts>(W);
  C.addTransition(State);
}

void ento::registerPlacementSelfCopyChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<PlacementSelfCopyChecker>();
}

bool ento::shouldRegisterPlacementSelfCopyChecker(const CheckerManager &) {
  return true;
}
