//===----- UninitializedPointee.cpp ------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions and methods for handling pointers and references
// to reduce the size and complexity of UninitializedObjectChecker.cpp.
//
// To read about command line options and documentation about how the checker
// works, refer to UninitializedObjectChecker.h.
//
//===----------------------------------------------------------------------===//

#include "../ClangSACheckers.h"
#include "UninitializedObject.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeMap.h"

using namespace clang;
using namespace clang::ento;

namespace {

/// Represents a pointer or a reference field.
class LocField final : public FieldNode {
  /// We'll store whether the pointee or the pointer itself is uninitialited.
  const bool IsDereferenced;

public:
  LocField(const FieldRegion *FR, const bool IsDereferenced = true)
      : FieldNode(FR), IsDereferenced(IsDereferenced) {}

  virtual void printNoteMsg(llvm::raw_ostream &Out) const override {
    if (IsDereferenced)
      Out << "uninitialized pointee ";
    else
      Out << "uninitialized pointer ";
  }

  virtual void printPrefix(llvm::raw_ostream &Out) const override {}

  virtual void printNode(llvm::raw_ostream &Out) const override {
    Out << getVariableName(getDecl());
  }

  virtual void printSeparator(llvm::raw_ostream &Out) const override {
    if (getDecl()->getType()->isPointerType())
      Out << "->";
    else
      Out << '.';
  }
};

/// Represents a void* field that needs to be casted back to its dynamic type
/// for a correct note message.
class NeedsCastLocField final : public FieldNode {
  QualType CastBackType;

public:
  NeedsCastLocField(const FieldRegion *FR, const QualType &T)
      : FieldNode(FR), CastBackType(T) {}

  virtual void printNoteMsg(llvm::raw_ostream &Out) const override {
    Out << "uninitialized pointee ";
  }

  virtual void printPrefix(llvm::raw_ostream &Out) const override {
    Out << "static_cast" << '<' << CastBackType.getAsString() << ">(";
  }

  virtual void printNode(llvm::raw_ostream &Out) const override {
    Out << getVariableName(getDecl()) << ')';
  }

  virtual void printSeparator(llvm::raw_ostream &Out) const override {
    Out << "->";
  }
};

} // end of anonymous namespace

// Utility function declarations.

/// Returns whether \p T can be (transitively) dereferenced to a void pointer
/// type (void*, void**, ...).
static bool isVoidPointer(QualType T);

using DereferenceInfo = std::pair<const TypedValueRegion *, bool>;

/// Dereferences \p FR and returns with the pointee's region, and whether it
/// needs to be casted back to it's location type. If for whatever reason
/// dereferencing fails, returns with None.
static llvm::Optional<DereferenceInfo> dereference(ProgramStateRef State,
                                                   const FieldRegion *FR);

//===----------------------------------------------------------------------===//
//                   Methods for FindUninitializedFields.
//===----------------------------------------------------------------------===//

bool FindUninitializedFields::isDereferencableUninit(
    const FieldRegion *FR, FieldChainInfo LocalChain) {

  assert(isDereferencableType(FR->getDecl()->getType()) &&
         "This method only checks dereferencable objects!");

  SVal V = State->getSVal(FR);

  if (V.isUnknown() || V.getAs<loc::ConcreteInt>()) {
    IsAnyFieldInitialized = true;
    return false;
  }

  if (V.isUndef()) {
    return addFieldToUninits(
        LocalChain.add(LocField(FR, /*IsDereferenced*/ false)));
  }

  if (!Opts.CheckPointeeInitialization) {
    IsAnyFieldInitialized = true;
    return false;
  }

  // At this point the pointer itself is initialized and points to a valid
  // location, we'll now check the pointee.
  llvm::Optional<DereferenceInfo> DerefInfo = dereference(State, FR);
  if (!DerefInfo) {
    IsAnyFieldInitialized = true;
    return false;
  }

  const TypedValueRegion *R = DerefInfo->first;
  const bool NeedsCastBack = DerefInfo->second;

  QualType DynT = R->getLocationType();
  QualType PointeeT = DynT->getPointeeType();

  if (PointeeT->isStructureOrClassType()) {
    if (NeedsCastBack)
      return isNonUnionUninit(R, LocalChain.add(NeedsCastLocField(FR, DynT)));
    return isNonUnionUninit(R, LocalChain.add(LocField(FR)));
  }

  if (PointeeT->isUnionType()) {
    if (isUnionUninit(R)) {
      if (NeedsCastBack)
        return addFieldToUninits(LocalChain.add(NeedsCastLocField(FR, DynT)));
      return addFieldToUninits(LocalChain.add(LocField(FR)));
    } else {
      IsAnyFieldInitialized = true;
      return false;
    }
  }

  if (PointeeT->isArrayType()) {
    IsAnyFieldInitialized = true;
    return false;
  }

  assert((isPrimitiveType(PointeeT) || isDereferencableType(PointeeT)) &&
         "At this point FR must either have a primitive dynamic type, or it "
         "must be a null, undefined, unknown or concrete pointer!");

  SVal PointeeV = State->getSVal(R);

  if (isPrimitiveUninit(PointeeV)) {
    if (NeedsCastBack)
      return addFieldToUninits(LocalChain.add(NeedsCastLocField(FR, DynT)));
    return addFieldToUninits(LocalChain.add(LocField(FR)));
  }

  IsAnyFieldInitialized = true;
  return false;
}

//===----------------------------------------------------------------------===//
//                           Utility functions.
//===----------------------------------------------------------------------===//

static bool isVoidPointer(QualType T) {
  while (!T.isNull()) {
    if (T->isVoidPointerType())
      return true;
    T = T->getPointeeType();
  }
  return false;
}

static llvm::Optional<DereferenceInfo> dereference(ProgramStateRef State,
                                                   const FieldRegion *FR) {

  llvm::SmallSet<const TypedValueRegion *, 5> VisitedRegions;

  // If the static type of the field is a void pointer, we need to cast it back
  // to the dynamic type before dereferencing.
  bool NeedsCastBack = isVoidPointer(FR->getDecl()->getType());

  SVal V = State->getSVal(FR);
  assert(V.getAsRegion() && "V must have an underlying region!");

  // The region we'd like to acquire.
  const auto *R = V.getAsRegion()->getAs<TypedValueRegion>();
  if (!R)
    return None;

  VisitedRegions.insert(R);

  // We acquire the dynamic type of R,
  QualType DynT = R->getLocationType();

  while (const MemRegion *Tmp = State->getSVal(R, DynT).getAsRegion()) {

    R = Tmp->getAs<TypedValueRegion>();
    if (!R)
      return None;

    // We found a cyclic pointer, like int *ptr = (int *)&ptr.
    // TODO: Should we report these fields too?
    if (!VisitedRegions.insert(R).second)
      return None;

    DynT = R->getLocationType();
    // In order to ensure that this loop terminates, we're also checking the
    // dynamic type of R, since type hierarchy is finite.
    if (isDereferencableType(DynT->getPointeeType()))
      break;
  }

  while (R->getAs<CXXBaseObjectRegion>()) {
    NeedsCastBack = true;

    if (!isa<TypedValueRegion>(R->getSuperRegion()))
      break;
    R = R->getSuperRegion()->getAs<TypedValueRegion>();
  }

  return std::make_pair(R, NeedsCastBack);
}
