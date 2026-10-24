//===- PlacementSelfCopyChecker.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker that finds placement new expressions that access the
// target storage through the old object while the new object is being
// constructed, e.g. `::new (p) T(*p)`. A call that is not inlined is assumed
// to access the old object passed to it; when it is the copy or move
// constructor called by placement new itself, it is reported as a self-copy.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/Builtins.h"
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
// Variables that refer to the old storage, mapped to the placement target.
REGISTER_MAP_WITH_PROGRAMSTATE(SourceVars, const MemRegion *, const MemRegion *)
// Frames of methods called on the old storage, mapped to the placement target.
REGISTER_MAP_WITH_PROGRAMSTATE(SourceFrames, const StackFrame *,
                               const MemRegion *)
REGISTER_TRAIT_WITH_PROGRAMSTATE(PendingSelfCopy, const MemRegion *)

namespace {
class PlacementSelfCopyChecker
    : public Checker<check::PostCall, check::PreCall, check::Bind,
                     check::Location, check::PostStmt<LambdaExpr>,
                     check::PostStmt<CXXNewExpr>> {
  const BugType BT{this, "Placement new initialized from its own storage",
                   categories::MemoryError};
  const BugType AccessBT{this, "Placement new accesses its own storage",
                         categories::MemoryError};

  void reportAccess(const Expr *E, CheckerContext &C) const;
  void reportSelfCopy(const Expr *Arg, CheckerContext &C) const;
  void reportPendingSelfCopy(const CallEvent &Call, CheckerContext &C) const;

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkBind(SVal Loc, SVal Val, const Stmt *S, bool AtDeclInit,
                 CheckerContext &C) const;
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const;
  void checkPostStmt(const LambdaExpr *LE, CheckerContext &C) const;
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

// Base class subobjects keep their own extent.
static const MemRegion *stripCasts(const MemRegion *R) {
  return R->StripCasts(/*StripBaseAndDerivedCasts=*/false);
}

static std::optional<Extent> getExtent(const MemRegion *R, ASTContext &Ctx) {
  RegionOffset Offset = R->getAsOffset();
  if (!Offset.isValid() || Offset.hasSymbolicOffset())
    return std::nullopt;
  Extent E{Offset.getRegion(), Offset.getOffset(), std::nullopt};
  QualType Ty;
  if (const auto *TR = dyn_cast<TypedValueRegion>(R))
    Ty = TR->getValueType();
  else if (const auto *SR = dyn_cast<SymbolicRegion>(R))
    Ty = SR->getPointeeStaticType();
  if (!Ty.isNull() && !Ty->isIncompleteType())
    E.End = E.Begin + Ctx.getTypeSize(Ty);
  return E;
}

// Returns the extents of A and B if both are known within the same base.
static std::optional<std::pair<Extent, Extent>>
getKnownExtents(const MemRegion *A, const MemRegion *B, ASTContext &Ctx) {
  std::optional<Extent> EA = getExtent(A, Ctx), EB = getExtent(B, Ctx);
  if (!EA || !EB || !EA->End || !EB->End || EA->Base != EB->Base)
    return std::nullopt;
  return std::make_pair(*EA, *EB);
}

static bool overlaps(const MemRegion *A, const MemRegion *B, ASTContext &Ctx) {
  if (auto E = getKnownExtents(A, B, Ctx))
    return E->first.Begin < *E->second.End && E->second.Begin < *E->first.End;
  A = stripCasts(A);
  return A == stripCasts(B) || A->isSubRegionOf(B) || B->isSubRegionOf(A);
}

static bool isWithin(const MemRegion *R, const MemRegion *Container,
                     ASTContext &Ctx) {
  if (auto E = getKnownExtents(R, Container, Ctx))
    return E->second.Begin <= E->first.Begin && *E->first.End <= *E->second.End;
  R = stripCasts(R);
  return R == stripCasts(Container) || R->isSubRegionOf(Container);
}

// Limits R to the placed object instead of the whole buffer it lives in.
static const MemRegion *getPlacedRegion(const CXXNewExpr *NE,
                                        const MemRegion *R, CheckerContext &C) {
  ASTContext &Ctx = C.getASTContext();
  QualType Ty = NE->getAllocatedType();
  if (NE->isArray()) {
    std::optional<const Expr *> Size = NE->getArraySize();
    std::optional<llvm::APSInt> N =
        Size && *Size ? (*Size)->getIntegerConstantExpr(Ctx) : std::nullopt;
    if (!N)
      return R;
    Ty =
        Ctx.getConstantArrayType(Ty, *N, nullptr, ArraySizeModifier::Normal, 0);
  }
  const auto *SR = dyn_cast<SubRegion>(R);
  const auto *TR = dyn_cast<TypedValueRegion>(R);
  if (!SR || (TR && Ctx.hasSameUnqualifiedType(TR->getValueType(), Ty)))
    return R;
  return C.getStoreManager().getRegionManager().getElementRegion(
      Ty, C.getSValBuilder().makeArrayIndex(0), SR, Ctx);
}

// Calls that return their argument or its address without accessing it, e.g.
// `std::move(X)`, `std::addressof(X)` or `&X` with an overloaded `operator&`.
static bool isPassThroughCall(const CallExpr *CE) {
  if (!CE || CE->getNumArgs() != 1)
    return false;
  if (const auto *OCE = dyn_cast<CXXOperatorCallExpr>(CE))
    return OCE->getOperator() == OO_Amp;
  const FunctionDecl *FD = CE->getDirectCallee();
  switch (FD ? FD->getBuiltinID() : 0) {
  case Builtin::BIaddressof:
  case Builtin::BI__addressof:
  case Builtin::BI__builtin_addressof:
  case Builtin::BIas_const:
  case Builtin::BIforward:
  case Builtin::BIforward_like:
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
    return true;
  default:
    return false;
  }
}

// Collects the variables that E accesses memory through, e.g. `O` for
// `O->X[1]`, `O.A + 1` or `O.data()`, and both `A` and `B` for
// `(C ? A : B).X`.
static void collectBaseVars(const Expr *E,
                            SmallVectorImpl<const VarDecl *> &Vars,
                            bool &ViaThis) {
  while (E) {
    E = E->IgnoreParenCasts();
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        Vars.push_back(VD);
      return;
    }
    if (isa<CXXThisExpr>(E)) {
      ViaThis = true;
      return;
    }
    if (const auto *CO = dyn_cast<AbstractConditionalOperator>(E)) {
      collectBaseVars(CO->getTrueExpr(), Vars, ViaThis);
      E = CO->getFalseExpr();
    } else if (const auto *ME = dyn_cast<MemberExpr>(E))
      E = ME->getBase();
    else if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E))
      E = ASE->getBase();
    else if (const auto *BO = dyn_cast<BinaryOperator>(E);
             BO && BO->isAssignmentOp())
      E = BO->getLHS();
    else if (BO && BO->isAdditiveOp())
      E = BO->getLHS()->getType()->isPointerType() ? BO->getLHS()
                                                   : BO->getRHS();
    else if (const auto *UO = dyn_cast<UnaryOperator>(E);
             UO &&
             (UO->getOpcode() == UO_Deref || UO->getOpcode() == UO_AddrOf ||
              UO->isIncrementDecrementOp()))
      E = UO->getSubExpr();
    else if (const auto *CE = dyn_cast<CallExpr>(E); isPassThroughCall(CE))
      E = CE->getArg(0);
    else if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(E))
      E = MCE->getImplicitObjectArgument();
    else
      return;
  }
}

static bool isEmptyRecord(const MemRegion *R) {
  const auto *TR = dyn_cast<TypedValueRegion>(R);
  const CXXRecordDecl *RD =
      TR ? TR->getValueType()->getAsCXXRecordDecl() : nullptr;
  return RD && RD->hasDefinition() && RD->isEmpty();
}

static bool isInInitializer(const CXXNewExpr *NE, const Stmt *S,
                            CheckerContext &C) {
  const Expr *Init = NE->getInitializer();
  const ParentMap &PM = C.getCurrentAnalysisDeclContext()->getParentMap();
  for (const Stmt *P = S; P && P != NE; P = PM.getParent(P))
    if (P == Init)
      return true;
  return false;
}

// Returns the target whose new-initializer S uses Source in.
static const MemRegion *findInitializedTarget(const Stmt *S,
                                              const MemRegion *Source,
                                              CheckerContext &C) {
  if (isEmptyRecord(Source))
    return nullptr;
  for (const auto &[Key, Target] : C.getState()->get<PlacementTargets>())
    if (Key.second == C.getStackFrame() && isInInitializer(Key.first, S, C) &&
        overlaps(Source, Target, C.getASTContext()))
      return Target;
  return nullptr;
}

// Captured variables and `this` in a lambda body belong to the frame of the
// function that defines the lambda. Only pointers and `this` get here: other
// captures of the old object are reported when the lambda is created.
static const StackFrame *getOwningFrame(const VarDecl *VD,
                                        const StackFrame *Frame) {
  while (Frame) {
    const auto *MD = dyn_cast_if_present<CXXMethodDecl>(Frame->getDecl());
    if (!MD || !MD->getParent()->isLambda() ||
        (VD && VD->getDeclContext() == MD))
      return Frame;
    const auto *Outer = dyn_cast<Decl>(MD->getParent()->getDeclContext());
    do
      Frame = Frame->getParent();
    while (Frame && Frame->getDecl() != Outer);
  }
  return nullptr;
}

// Returns the target that E accesses through a variable that refers to it,
// or through `this` of a method called on the old storage.
static const MemRegion *findSourceTarget(const Expr *E, const MemRegion *Source,
                                         CheckerContext &C) {
  if (isEmptyRecord(Source))
    return nullptr;
  SmallVector<const VarDecl *, 2> Vars;
  bool ViaThis = false;
  collectBaseVars(E, Vars, ViaThis);
  ProgramStateRef State = C.getState();
  auto Matches = [&](const MemRegion *const *Target) {
    return Target && overlaps(Source, *Target, C.getASTContext());
  };
  for (const VarDecl *VD : Vars) {
    const StackFrame *Owner = getOwningFrame(VD, C.getStackFrame());
    const MemRegion *Var =
        Owner ? State->getLValue(VD, Owner).getAsRegion() : nullptr;
    if (const MemRegion *const *T = Var ? State->get<SourceVars>(Var) : nullptr;
        Matches(T))
      return *T;
  }
  const StackFrame *Owner =
      ViaThis ? getOwningFrame(nullptr, C.getStackFrame()) : nullptr;
  if (const MemRegion *const *T =
          Owner ? State->get<SourceFrames>(Owner) : nullptr;
      Matches(T))
    return *T;
  return nullptr;
}

// Returns the target that E refers to in call S, either through the old
// storage or directly in the new-initializer.
static const MemRegion *findTarget(const Expr *E, const MemRegion *Source,
                                   const Stmt *S, CheckerContext &C) {
  if (const MemRegion *Target = findSourceTarget(E, Source, C))
    return Target;
  return findInitializedTarget(S, Source, C);
}

// Returns whether E refers to a part of the old storage, not to an object that
// merely contains it, such as a pool with an inline buffer.
static bool refersToOldObject(const Expr *E, const MemRegion *Source,
                              const Stmt *S, CheckerContext &C) {
  const MemRegion *Target = findTarget(E, Source, S, C);
  if (!Target || !isWithin(Source, Target, C.getASTContext()))
    return false;
  // An enclosing object of the same size, e.g. `O->Inner` for a target `Inner`.
  return stripCasts(Target) == stripCasts(Source) ||
         !Target->isSubRegionOf(Source);
}

// Returns the argument that passes the old object to a call that is not
// inlined. Such a call may access it in any way.
static const Expr *findOpaqueSourceUse(const CallEvent &Call,
                                       CheckerContext &C) {
  if (isPassThroughCall(dyn_cast_if_present<CallExpr>(Call.getOriginExpr())))
    return nullptr;
  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    const Expr *Obj = IC->getCXXThisExpr();
    const MemRegion *Source = IC->getCXXThisVal().getAsRegion();
    if (Obj && Source &&
        refersToOldObject(Obj, Source, Call.getOriginExpr(), C))
      return Obj;
  }
  for (unsigned I = 0, E = Call.getNumArgs(); I != E; ++I) {
    const MemRegion *Source = Call.getArgSVal(I).getAsRegion();
    const Expr *Arg = Call.getArgExpr(I);
    if (Source && Arg &&
        refersToOldObject(Arg, Source, Call.getOriginExpr(), C))
      return Arg;
  }
  return nullptr;
}

void PlacementSelfCopyChecker::reportAccess(const Expr *E,
                                            CheckerContext &C) const {
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  auto R = std::make_unique<PathSensitiveBugReport>(
      AccessBT,
      "Placement new accesses the storage it is placed into during "
      "construction of the new object",
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
  const Expr *Origin = Call.getOriginExpr();
  if (!Target || !Origin ||
      C.getSourceManager().isInSystemHeader(Origin->getBeginLoc()))
    return;
  for (unsigned I = 0, E = Call.getNumArgs(); I != E; ++I) {
    const MemRegion *Arg = Call.getArgSVal(I).getAsRegion();
    if (Arg && !Call.getArgExpr(I)->getType()->isPointerType() &&
        overlaps(Arg, Target, C.getASTContext())) {
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
    // The operator may be declared without a body.
    const MemRegion *Target =
        Alloc->getArgSVal(Alloc->getNumImplicitArgs()).getAsRegion();
    if (!Target)
      return;
    Target = getPlacedRegion(NE, stripCasts(Target), C);
    C.addTransition(
        C.getState()->set<PlacementTargets>({NE, C.getStackFrame()}, Target));
    return;
  }

  // Inlined calls are checked per read.
  const auto *Ctor = dyn_cast<CXXConstructorCall>(&Call);
  if (!C.wasInlined && !C.getState()->get<PlacementTargets>().isEmpty()) {
    const CXXConstructorDecl *CD = Ctor ? Ctor->getDecl() : nullptr;
    const MemRegion *Source = CD && CD->isCopyOrMoveConstructor()
                                  ? Ctor->getArgSVal(0).getAsRegion()
                                  : nullptr;
    if (const MemRegion *Target =
            Source ? findInitializedTarget(Ctor->getOriginExpr(), Source, C)
                   : nullptr) {
      if (C.getSourceManager().isInSystemHeader(
              Ctor->getOriginExpr()->getBeginLoc()))
        C.addTransition(C.getState()->set<PendingSelfCopy>(Target));
      else
        reportSelfCopy(Ctor->getArgExpr(0), C);
      return;
    }
    if (const Expr *E = findOpaqueSourceUse(Call, C)) {
      reportAccess(E, C);
      return;
    }
  }
  if (!Ctor)
    reportPendingSelfCopy(Call, C);
}

void PlacementSelfCopyChecker::checkPreCall(const CallEvent &Call,
                                            CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  if (State->get<PlacementTargets>().isEmpty())
    return;
  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    const Expr *Obj = IC->getCXXThisExpr();
    const MemRegion *Source = IC->getCXXThisVal().getAsRegion();
    const MemRegion *Target =
        Obj && Source ? findTarget(Obj, Source, Call.getOriginExpr(), C)
                      : nullptr;
    if (const StackFrame *Frame =
            Target ? Call.getCalleeStackFrame(C.blockCount()) : nullptr)
      State = State->set<SourceFrames>(Frame, Target);
  }
  ArrayRef<ParmVarDecl *> Params = Call.parameters();
  unsigned NumArgs = std::min<size_t>(Call.getNumArgs(), Params.size());
  for (unsigned I = 0; I != NumArgs; ++I) {
    QualType ParamTy = Params[I]->getType();
    const MemRegion *Source = Call.getArgSVal(I).getAsRegion();
    const Expr *Arg = Call.getArgExpr(I);
    if ((!ParamTy->isReferenceType() && !ParamTy->isPointerType()) || !Source ||
        !Arg)
      continue;
    const MemRegion *Target = findTarget(Arg, Source, Call.getOriginExpr(), C);
    if (const ParamVarRegion *Param =
            Target ? Call.getParameterLocation(I, C.blockCount()) : nullptr)
      State = State->set<SourceVars>(Param, Target);
  }
  C.addTransition(State);
}

void PlacementSelfCopyChecker::checkBind(SVal Loc, SVal Val, const Stmt *S,
                                         bool AtDeclInit,
                                         CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const auto *Var = dyn_cast_or_null<VarRegion>(Loc.getAsRegion());
  if (!Var || State->get<PlacementTargets>().isEmpty())
    return;
  QualType Ty = Var->getDecl()->getType();
  if (!Ty->isReferenceType() && !Ty->isPointerType())
    return;
  const Expr *Init = nullptr;
  if (AtDeclInit)
    Init = Var->getDecl()->getInit();
  else if (const auto *BO = dyn_cast_if_present<BinaryOperator>(S);
           BO && BO->getOpcode() == BO_Assign)
    Init = BO->getRHS();
  else
    Init = dyn_cast_if_present<Expr>(S);
  const MemRegion *Source = Val.getAsRegion();
  const MemRegion *Target =
      Init && Source ? findSourceTarget(Init, Source, C) : nullptr;
  if (Target)
    C.addTransition(State->set<SourceVars>(Var, Target));
  else if (State->contains<SourceVars>(Var))
    C.addTransition(State->remove<SourceVars>(Var));
}

void PlacementSelfCopyChecker::checkLocation(SVal Loc, bool IsLoad,
                                             const Stmt *S,
                                             CheckerContext &C) const {
  if (C.getState()->get<PlacementTargets>().isEmpty())
    return;
  const auto *E = dyn_cast_if_present<Expr>(S);
  const MemRegion *R = Loc.getAsRegion();
  if (!E || !R)
    return;
  if (findTarget(E, R, S, C))
    reportAccess(E, C);
}

// A lambda that captures the old object reads or modifies it.
void PlacementSelfCopyChecker::checkPostStmt(const LambdaExpr *LE,
                                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  if (State->get<SourceVars>().isEmpty())
    return;
  for (const LambdaCapture &Capture : LE->captures()) {
    const auto *VD = Capture.capturesVariable()
                         ? dyn_cast<VarDecl>(Capture.getCapturedVar())
                         : nullptr;
    if (!VD || VD->getType()->isPointerType())
      continue;
    const MemRegion *Var =
        State->getLValue(VD, C.getStackFrame()).getAsRegion();
    const MemRegion *const *Target =
        Var ? State->get<SourceVars>(Var) : nullptr;
    const MemRegion *Source =
        Target ? State->getSVal(Var).getAsRegion() : nullptr;
    if (Source && isWithin(Source, *Target, C.getASTContext())) {
      reportAccess(LE, C);
      return;
    }
  }
}

void PlacementSelfCopyChecker::checkPostStmt(const CXXNewExpr *NE,
                                             CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  PlacementKey Key{NE, C.getStackFrame()};
  const MemRegion *const *Target = State->get<PlacementTargets>(Key);
  if (!Target)
    return;
  State = State->remove<PlacementTargets>(Key);
  for (const auto &[Var, VarTarget] : State->get<SourceVars>())
    if (VarTarget == *Target)
      State = State->remove<SourceVars>(Var);
  for (const auto &[Frame, FrameTarget] : State->get<SourceFrames>())
    if (FrameTarget == *Target)
      State = State->remove<SourceFrames>(Frame);
  C.addTransition(State);
}

void ento::registerPlacementSelfCopyChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<PlacementSelfCopyChecker>();
}

bool ento::shouldRegisterPlacementSelfCopyChecker(const CheckerManager &) {
  return true;
}
