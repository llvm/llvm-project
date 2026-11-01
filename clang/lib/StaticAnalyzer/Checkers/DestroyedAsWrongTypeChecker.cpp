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
};
} // namespace

REGISTER_MAP_WITH_PROGRAMSTATE(PlacedObjectTypes, const MemRegion *, QualType)
REGISTER_SET_WITH_PROGRAMSTATE(NewAllocatedObjects, const MemRegion *)
REGISTER_MAP_WITH_PROGRAMSTATE(PendingDestructions, const MemRegion *,
                               WrongDestruction)

namespace {
class DestroyedAsWrongTypeChecker
    : public Checker<check::PostStmt<CXXNewExpr>, check::PreCall,
                     check::PostCall, check::DeadSymbols> {
  const BugType BT{this, "Object destroyed as a type it no longer has",
                   categories::MemoryError};

  void reportBug(const MemRegion *R, const WrongDestruction &W,
                 CheckerContext &C) const;
  void reportOrDefer(const MemRegion *R, const WrongDestruction &W,
                     CheckerContext &C) const;

public:
  void checkPostStmt(const CXXNewExpr *NE, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
};
} // namespace

// An array is keyed by its first element, the object its destructor gets.
static const MemRegion *getStorageRegion(SVal V, CheckerContext &C) {
  const MemRegion *R = V.getAsRegion();
  if (!R)
    return nullptr;
  R = R->StripCasts();
  ASTContext &Ctx = C.getASTContext();
  while (const auto *TR = dyn_cast<TypedValueRegion>(R)) {
    const ArrayType *AT = Ctx.getAsArrayType(TR->getValueType());
    if (!AT)
      break;
    R = C.getStoreManager().getRegionManager().getElementRegion(
        AT->getElementType(), C.getSValBuilder().makeZeroArrayIndex(), TR, Ctx);
  }
  return R;
}

// Storage obtained as raw memory has no original object to be replaced.
static bool holdsOriginalObject(ProgramStateRef State, const MemRegion *R) {
  if (State->contains<NewAllocatedObjects>(R))
    return true;
  const auto *TR = dyn_cast<TypedValueRegion>(R);
  if (!TR)
    return false;
  // Placement new into a union starts the lifetime of one of its members.
  const CXXRecordDecl *RD = TR->getValueType()->getAsCXXRecordDecl();
  return RD && !RD->isUnion() && RD->hasNonTrivialDestructor();
}

static bool isSameRecord(QualType T, const CXXRecordDecl *RD) {
  const CXXRecordDecl *TD = T->getAsCXXRecordDecl();
  return TD && TD->getCanonicalDecl() == RD->getCanonicalDecl();
}

static QualType getStorageType(const MemRegion *R) {
  if (const auto *TR = dyn_cast<TypedValueRegion>(R))
    return TR->getValueType();
  if (const auto *SR = dyn_cast<SymbolicRegion>(R))
    return SR->getPointeeStaticType();
  return {};
}

// A field and a byte offset from its parent are different regions.
static bool isSameLocation(const MemRegion *A, const MemRegion *B) {
  RegionOffset OA = A->getAsOffset();
  RegionOffset OB = B->getAsOffset();
  return OA.isValid() && OB.isValid() && !OA.hasSymbolicOffset() &&
         !OB.hasSymbolicOffset() && OA.getRegion() == OB.getRegion() &&
         OA.getOffset() == OB.getOffset();
}

static ProgramStateRef forgetPlacedTypes(ProgramStateRef State,
                                         const MemRegion *R) {
  for (const MemRegion *Key :
       llvm::make_first_range(State->get<PlacedObjectTypes>()))
    if (Key == R || Key->isSubRegionOf(R) || isSameLocation(Key, R))
      State = State->remove<PlacedObjectTypes>(Key);
  return State;
}

static SmallVector<const MemRegion *, 1>
getPlacementStorages(const CXXNewExpr *NE, CheckerContext &C) {
  SmallVector<const MemRegion *, 1> Storages;
  for (const Expr *Arg : NE->placement_arguments())
    if (Arg->getType()->isPointerType())
      if (const MemRegion *R = getStorageRegion(C.getSVal(Arg), C))
        Storages.push_back(R);
  return Storages;
}

static bool isInStdFrame(CheckerContext &C) {
  return AnalysisDeclContext::isInStdNamespace(C.getStackFrame()->getDecl());
}

void DestroyedAsWrongTypeChecker::reportBug(const MemRegion *R,
                                            const WrongDestruction &W,
                                            CheckerContext &C) const {
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;
  const PrintingPolicy &PP = C.getASTContext().getPrintingPolicy();
  std::string Placed = W.Placed.getAsString(PP);
  std::string Msg =
      W.Delete
          ? "Deleting an object of type '" + Placed +
                "' through a pointer to '" +
                W.Delete->getDestroyedType().getUnqualifiedType().getAsString(
                    PP) +
                "'"
          : "Destructor of '" + W.DtorClass->getQualifiedNameAsString() +
                "' is called on storage that holds an object of type '" +
                Placed + "'";
  auto Report = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
  Report->markInteresting(R);
  C.emitReport(std::move(Report));
}

void DestroyedAsWrongTypeChecker::checkPostStmt(const CXXNewExpr *NE,
                                                CheckerContext &C) const {
  const FunctionDecl *OperatorNew = NE->getOperatorNew();
  if (!OperatorNew)
    return;

  const MemRegion *Result = getStorageRegion(C.getSVal(NE), C);
  if (!Result)
    return;

  ProgramStateRef State = C.getState();
  SmallVector<const MemRegion *, 1> Storages = getPlacementStorages(NE, C);
  if (OperatorNew->isReservedGlobalPlacementOperator() && Storages.empty())
    return;
  // The result may be typed as the new object, so use the storage argument.
  const MemRegion *R = nullptr;
  for (const MemRegion *Storage : Storages)
    if (OperatorNew->isReservedGlobalPlacementOperator() || Storage == Result ||
        isSameLocation(Storage, Result))
      R = Storage;
  if (!R) {
    // An operator new that is not inlined may have placed the object there.
    for (const MemRegion *Storage : Storages)
      State = forgetPlacedTypes(State, Storage);
    C.addTransition(State->add<NewAllocatedObjects>(Result));
    return;
  }

  if (NE->isArray()) {
    const auto *ER = dyn_cast<ElementRegion>(R);
    C.addTransition(forgetPlacedTypes(State, ER ? ER->getSuperRegion() : R));
    return;
  }
  State = forgetPlacedTypes(State, R);
  if (!holdsOriginalObject(State, R)) {
    C.addTransition(State);
    return;
  }

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
  C.addTransition(State->set<PlacedObjectTypes>(R, Placed), Tag);
}

// Destroying through a base class with a virtual destructor is well-defined.
static bool isVirtualBaseDestruction(QualType Placed,
                                     const CXXRecordDecl *Static) {
  const CXXRecordDecl *PlacedRD = Placed->getAsCXXRecordDecl();
  if (!PlacedRD || !PlacedRD->isDerivedFrom(Static))
    return false;
  const CXXDestructorDecl *Dtor = Static->getDestructor();
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

static const MemRegion *getDestroyedRegion(const CallEvent &Call,
                                           CheckerContext &C) {
  const auto *Instance = dyn_cast<CXXInstanceCall>(&Call);
  return Instance ? getStorageRegion(Instance->getCXXThisVal(), C) : nullptr;
}

static std::optional<WrongDestruction>
findWrongDestruction(const CallEvent &Call, const CXXDestructorDecl *Dtor,
                     const MemRegion *R, ProgramStateRef State) {
  const QualType *Placed = State->get<PlacedObjectTypes>(R);
  const CXXRecordDecl *DtorClass = Dtor->getParent();
  if (!Placed || isSameRecord(*Placed, DtorClass))
    return std::nullopt;

  const auto *Implicit = dyn_cast<CXXDestructorCall>(&Call);
  std::optional<CFGDeleteDtor> Delete =
      Implicit ? Implicit->getCFGElementRef()->getAs<CFGDeleteDtor>()
               : std::nullopt;
  // Implicit destructors of automatic objects and members are not virtual.
  bool CallIsVirtual = !Implicit || Delete;
  if (CallIsVirtual && isVirtualBaseDestruction(*Placed, DtorClass))
    return std::nullopt;
  return WrongDestruction{*Placed, DtorClass,
                          Delete ? Delete->getDeleteExpr() : nullptr};
}

// A destructor without a body destroys members, and the analyzer then destroys
// only the last element of an array, out of the checker's sight.
static std::optional<std::pair<const MemRegion *, WrongDestruction>>
findHiddenWrongDestruction(const CallEvent &Call, const MemRegion *R,
                           ProgramStateRef State) {
  const MemRegion *Scope = R;
  if (const auto *ER = dyn_cast<ElementRegion>(R);
      ER && isa<CXXDestructorCall>(Call))
    Scope = ER->getSuperRegion();
  for (const auto &[Key, Placed] : State->get<PlacedObjectTypes>()) {
    QualType StorageType = getStorageType(Key);
    const CXXRecordDecl *RD =
        StorageType.isNull() ? nullptr : StorageType->getAsCXXRecordDecl();
    if (RD && Key != R && Key->isSubRegionOf(Scope) &&
        !isSameRecord(Placed, RD))
      return std::make_pair(Key, WrongDestruction{Placed, RD, nullptr});
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

  const MemRegion *R = getDestroyedRegion(Call, C);
  if (!R)
    return;
  ProgramStateRef State = C.getState();
  if (std::optional<WrongDestruction> W =
          findWrongDestruction(Call, Dtor, R, State)) {
    reportOrDefer(R, *W, C);
    return;
  }
  if (Call.getRuntimeDefinition().getDecl())
    return;
  if (auto Hidden = findHiddenWrongDestruction(Call, R, State))
    reportOrDefer(Hidden->first, Hidden->second, C);
}

void DestroyedAsWrongTypeChecker::checkPostCall(const CallEvent &Call,
                                                CheckerContext &C) const {
  PendingDestructionsTy Pending = C.getState()->get<PendingDestructions>();
  if (Pending.isEmpty() || isInStdFrame(C))
    return;
  const auto &[R, W] = *Pending.begin();
  reportBug(R, W, C);
}

void DestroyedAsWrongTypeChecker::checkDeadSymbols(SymbolReaper &SR,
                                                   CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  for (const MemRegion *R :
       llvm::make_first_range(State->get<PlacedObjectTypes>()))
    if (!SR.isLiveRegion(R))
      State = State->remove<PlacedObjectTypes>(R);
  for (const MemRegion *R : State->get<NewAllocatedObjects>())
    if (!SR.isLiveRegion(R))
      State = State->remove<NewAllocatedObjects>(R);
  C.addTransition(State);
}

void ento::registerDestroyedAsWrongTypeChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<DestroyedAsWrongTypeChecker>();
}

bool ento::shouldRegisterDestroyedAsWrongTypeChecker(const CheckerManager &) {
  return true;
}
