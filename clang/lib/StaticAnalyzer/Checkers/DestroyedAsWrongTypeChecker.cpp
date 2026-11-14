//===- DestroyedAsWrongTypeChecker.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker that finds storage reused by placement new for an object
// of a different type and then destroyed or deleted as the original type
// (C++ [basic.life]p11, [expr.delete]p3).
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
#include "clang/Analysis/CFG.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicType.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {
struct WrongDestruction {
  QualType Placed;
  const CXXRecordDecl *DtorClass;
  const CXXDeleteExpr *Delete;

  bool operator==(const WrongDestruction &X) const {
    return Placed == X.Placed && DtorClass == X.DtorClass && Delete == X.Delete;
  }
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.Add(Placed);
    ID.AddPointer(DtorClass);
    ID.AddPointer(Delete);
  }

  std::string getReportMessage(const PrintingPolicy &PP,
                               const NamedDecl *StdCaller) const {
    std::string Caller;
    if (StdCaller)
      Caller = " by '" + StdCaller->getQualifiedNameAsString() + "'";
    if (Delete)
      return llvm::formatv(
          "Deleting an object of type '{0}' through a pointer to '{1}'{2}",
          Placed.getAsString(PP),
          Delete->getDestroyedType().getUnqualifiedType().getAsString(PP),
          Caller);
    return llvm::formatv("Destructor of '{0}' is called{1} on storage that "
                         "holds an object of type '{2}'",
                         DtorClass->getQualifiedNameAsString(), Caller,
                         Placed.getAsString(PP));
  }
};
} // namespace

// Memory from a non-placement new, which holds an original object.
REGISTER_SET_WITH_PROGRAMSTATE(ObjectsAllocatedByNew, const MemRegion *)
// Storage of an original object reused by placement new.
REGISTER_SET_WITH_PROGRAMSTATE(ReusedStorages, const MemRegion *)
// Found inside std; reported on return to user code.
REGISTER_MAP_WITH_PROGRAMSTATE(PendingDestructions, const MemRegion *,
                               WrongDestruction)

namespace {
class DestroyedAsWrongTypeChecker
    : public Checker<check::PostStmt<CXXNewExpr>, check::PreCall,
                     check::PostCall, check::DeadSymbols> {
  const BugType BT{this, "Object destroyed as a type it no longer has",
                   categories::MemoryError};

  void reportBug(const MemRegion *R, const WrongDestruction &W,
                 CheckerContext &C, const NamedDecl *StdCaller = nullptr) const;
  void reportOrDefer(const MemRegion *R, const WrongDestruction &W,
                     CheckerContext &C) const;

public:
  void checkPostStmt(const CXXNewExpr *NE, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
};
} // namespace

// The type of the object known to occupy the storage, if it is exact.
static std::optional<QualType> getPlacedType(const DynamicTypeInfo &DTI) {
  if (!DTI.isValid() || DTI.canBeASubClass())
    return std::nullopt;
  QualType T = DTI.getType();
  if (T->isPointerType() || T->isReferenceType())
    T = T->getPointeeType();
  return T.getUnqualifiedType();
}

// Storage obtained as raw memory has no original object to be replaced.
static bool holdsOriginalObject(ProgramStateRef State, const MemRegion *R) {
  if (State->contains<ObjectsAllocatedByNew>(R))
    return true;
  const auto *TR = dyn_cast<TypedValueRegion>(R);
  if (!TR)
    return false;
  ASTContext &Ctx = State->getStateManager().getContext();
  // Placement new into a union starts the lifetime of one of its members.
  const CXXRecordDecl *RD =
      Ctx.getBaseElementType(TR->getValueType())->getAsCXXRecordDecl();
  return RD && !RD->isUnion() && RD->hasNonTrivialDestructor();
}

static bool isSameRecord(QualType T, const CXXRecordDecl *RD) {
  const CXXRecordDecl *TD = T->getAsCXXRecordDecl();
  return TD && TD->getCanonicalDecl() == RD->getCanonicalDecl();
}

static const CXXRecordDecl *getStorageRecord(const MemRegion *R,
                                             ASTContext &Ctx) {
  QualType T;
  if (const auto *TR = dyn_cast<TypedValueRegion>(R))
    T = TR->getValueType();
  else if (const auto *SR = dyn_cast<SymbolicRegion>(R))
    T = SR->getPointeeStaticType();
  if (T.isNull())
    return nullptr;
  return Ctx.getBaseElementType(T)->getAsCXXRecordDecl();
}

static bool isInStdFrame(CheckerContext &C) {
  return AnalysisDeclContext::isInStdNamespace(C.getStackFrame()->getDecl());
}

void DestroyedAsWrongTypeChecker::reportBug(const MemRegion *R,
                                            const WrongDestruction &W,
                                            CheckerContext &C,
                                            const NamedDecl *StdCaller) const {
  ExplodedNode *N = C.addSink();
  if (!N)
    return;
  std::string Msg =
      W.getReportMessage(C.getASTContext().getPrintingPolicy(), StdCaller);
  auto Report = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
  Report->markInteresting(R);
  C.emitReport(std::move(Report));
}

void DestroyedAsWrongTypeChecker::checkPostStmt(const CXXNewExpr *NE,
                                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const MemRegion *R = getPlacementNewStorage(State, NE, C.getStackFrame());
  if (!R) {
    if (const MemRegion *Result = C.getSVal(NE).getAsRegion())
      C.addTransition(State->add<ObjectsAllocatedByNew>(Result->StripCasts()));
    return;
  }
  if (NE->isArray() || !holdsOriginalObject(State, R))
    return;

  QualType Placed = NE->getAllocatedType().getUnqualifiedType();
  std::string Note = "Object of type '" +
                     Placed.getAsString(C.getASTContext().getPrintingPolicy()) +
                     "' is created here by placement new";
  const NoteTag *Tag =
      C.getNoteTag([this, R, Note](PathSensitiveBugReport &BR) -> std::string {
        if (&BR.getBugType() != &BT || !BR.isInteresting(R))
          return "";
        return Note;
      });
  C.addTransition(State->add<ReusedStorages>(R), Tag);
}

static bool isDerivedFrom(QualType Placed, const CXXRecordDecl *Base) {
  const CXXRecordDecl *PlacedRD = Placed->getAsCXXRecordDecl();
  return PlacedRD && PlacedRD->isDerivedFrom(Base);
}

static bool hasVirtualDestructor(const CXXRecordDecl *RD) {
  const CXXDestructorDecl *Dtor = RD->getDestructor();
  return Dtor && Dtor->isVirtual();
}

void DestroyedAsWrongTypeChecker::reportOrDefer(const MemRegion *R,
                                                const WrongDestruction &W,
                                                CheckerContext &C) const {
  // Reports inside the standard library are suppressed; report at the caller.
  if (isInStdFrame(C)) {
    C.addTransition(C.getState()->set<PendingDestructions>(R, W));
    return;
  }
  reportBug(R, W, C);
}

static const MemRegion *getDestroyedRegion(const CallEvent &Call) {
  const auto *Instance = dyn_cast<CXXInstanceCall>(&Call);
  if (!Instance)
    return nullptr;
  const MemRegion *R = Instance->getCXXThisVal().getAsRegion();
  if (!R)
    return nullptr;
  return R->StripCasts();
}

// The array of which R is an element, or R itself if it is an array.
static const MemRegion *getEnclosingArray(const MemRegion *R, ASTContext &Ctx) {
  if (const auto *ER = dyn_cast<ElementRegion>(R))
    return ER->getSuperRegion();
  const auto *TR = dyn_cast<TypedValueRegion>(R);
  if (!TR || !Ctx.getAsArrayType(TR->getValueType()))
    return nullptr;
  return R;
}

static std::optional<WrongDestruction>
findWrongDestruction(const CallEvent &Call, const CXXDestructorDecl *Dtor,
                     const MemRegion *R, ProgramStateRef State) {
  const DynamicTypeInfo *DTI = getRawDynamicTypeInfo(State, R);
  if (!DTI)
    return std::nullopt;
  std::optional<QualType> Placed = getPlacedType(*DTI);
  const CXXRecordDecl *DtorClass = Dtor->getParent();
  // An object placed at the address of a union may be its member.
  if (!Placed || DtorClass->isUnion() || isSameRecord(*Placed, DtorClass) ||
      !State->contains<ReusedStorages>(R))
    return std::nullopt;

  const auto *Implicit = dyn_cast<CXXDestructorCall>(&Call);
  // An explicit destructor call may name a base class ([class.dtor]p16).
  if (!Implicit) {
    if (isDerivedFrom(*Placed, DtorClass))
      return std::nullopt;
    return WrongDestruction{*Placed, DtorClass, nullptr};
  }

  std::optional<CFGDeleteDtor> Delete =
      Implicit->getCFGElementRef()->getAs<CFGDeleteDtor>();
  if (!Delete)
    return WrongDestruction{*Placed, DtorClass, nullptr};
  const CXXDeleteExpr *DE = Delete->getDeleteExpr();
  // [expr.delete]p3 does not cover a destroying operator delete, and allows a
  // base class with a virtual destructor only for a single object.
  const FunctionDecl *OperatorDelete = DE->getOperatorDelete();
  if (OperatorDelete && OperatorDelete->isDestroyingOperatorDelete())
    return std::nullopt;
  if (!DE->isArrayForm() && isDerivedFrom(*Placed, DtorClass) &&
      hasVirtualDestructor(DtorClass))
    return std::nullopt;
  return WrongDestruction{*Placed, DtorClass, DE};
}

// Members of a union are not destroyed with it.
static bool isInUnion(const MemRegion *R, const MemRegion *Scope) {
  while (R != Scope) {
    const auto *SR = dyn_cast<SubRegion>(R);
    if (!SR)
      return false;
    R = SR->getSuperRegion();
    const auto *TR = dyn_cast<TypedValueRegion>(R);
    if (TR && TR->getValueType()->isUnionType())
      return true;
  }
  return false;
}

// Finds a wrong destruction of storage within Scope other than R itself.
static std::optional<std::pair<const MemRegion *, WrongDestruction>>
findHiddenWrongDestruction(const MemRegion *R, const MemRegion *Scope,
                           ProgramStateRef State, ASTContext &Ctx) {
  for (const auto &[Key, DTI] : getDynamicTypeInfoWithin(State, Scope)) {
    std::optional<QualType> Placed = getPlacedType(DTI);
    const CXXRecordDecl *RD = getStorageRecord(Key, Ctx);
    if (Placed && RD && !RD->isUnion() && Key != R &&
        !isSameRecord(*Placed, RD) && State->contains<ReusedStorages>(Key) &&
        !isInUnion(Key, Scope))
      return std::make_pair(Key, WrongDestruction{*Placed, RD, nullptr});
  }
  return std::nullopt;
}

void DestroyedAsWrongTypeChecker::checkPreCall(const CallEvent &Call,
                                               CheckerContext &C) const {
  const auto *Dtor = dyn_cast_if_present<CXXDestructorDecl>(Call.getDecl());
  if (!Dtor)
    return;

  const auto *Implicit = dyn_cast<CXXDestructorCall>(&Call);
  if (Implicit && Implicit->isBaseDestructor())
    return;

  const MemRegion *R = getDestroyedRegion(Call);
  if (!R)
    return;
  ProgramStateRef State = C.getState();
  if (std::optional<WrongDestruction> W =
          findWrongDestruction(Call, Dtor, R, State)) {
    reportOrDefer(R, *W, C);
    return;
  }
  // The analyzer may destroy an array with a single call on its last element,
  // and does not show the destruction of members by a destructor without a
  // body, so check these here.
  const MemRegion *Scope = nullptr;
  if (Implicit)
    Scope = getEnclosingArray(R, C.getASTContext());
  if (!Scope) {
    if (Call.getRuntimeDefinition().getDecl())
      return;
    Scope = R;
  }
  if (auto Hidden =
          findHiddenWrongDestruction(R, Scope, State, C.getASTContext()))
    reportOrDefer(Hidden->first, Hidden->second, C);
}

void DestroyedAsWrongTypeChecker::checkPostCall(const CallEvent &Call,
                                                CheckerContext &C) const {
  PendingDestructionsTy Pending = C.getState()->get<PendingDestructions>();
  if (Pending.isEmpty() || isInStdFrame(C))
    return;
  const auto &[R, W] = *Pending.begin();
  reportBug(R, W, C, dyn_cast_if_present<NamedDecl>(Call.getDecl()));
}

void DestroyedAsWrongTypeChecker::checkDeadSymbols(SymbolReaper &SR,
                                                   CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  for (const MemRegion *R : State->get<ObjectsAllocatedByNew>())
    if (!SR.isLiveRegion(R))
      State = State->remove<ObjectsAllocatedByNew>(R);
  for (const MemRegion *R : State->get<ReusedStorages>())
    if (!SR.isLiveRegion(R))
      State = State->remove<ReusedStorages>(R);
  C.addTransition(State);
}

void ento::registerDestroyedAsWrongTypeChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<DestroyedAsWrongTypeChecker>();
}

bool ento::shouldRegisterDestroyedAsWrongTypeChecker(
    const CheckerManager &Mgr) {
  return Mgr.getLangOpts().CPlusPlus;
}
