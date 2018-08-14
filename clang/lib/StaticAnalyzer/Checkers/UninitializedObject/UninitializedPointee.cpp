//===----- UninitializedPointer.cpp ------------------------------*- C++ -*-==//
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
// To read about command line options and a description what this checker does,
// refer to UninitializedObjectChecker.cpp.
//
// To read about how the checker works, refer to the comments in
// UninitializedObject.h.
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

} // end of anonymous namespace

// Utility function declarations.

/// Returns whether T can be (transitively) dereferenced to a void pointer type
/// (void*, void**, ...). The type of the region behind a void pointer isn't
/// known, and thus FD can not be analyzed.
static bool isVoidPointer(QualType T);

//===----------------------------------------------------------------------===//
//                   Methods for FindUninitializedFields.
//===----------------------------------------------------------------------===//

// Note that pointers/references don't contain fields themselves, so in this
// function we won't add anything to LocalChain.
bool FindUninitializedFields::isPointerOrReferenceUninit(
    const FieldRegion *FR, FieldChainInfo LocalChain) {

  assert((FR->getDecl()->getType()->isAnyPointerType() ||
          FR->getDecl()->getType()->isReferenceType() ||
          FR->getDecl()->getType()->isBlockPointerType()) &&
         "This method only checks pointer/reference objects!");

  SVal V = State->getSVal(FR);

  if (V.isUnknown() || V.getAs<loc::ConcreteInt>()) {
    IsAnyFieldInitialized = true;
    return false;
  }

  if (V.isUndef()) {
    return addFieldToUninits(
        LocalChain.add(LocField(FR, /*IsDereferenced*/ false)));
  }

  if (!CheckPointeeInitialization) {
    IsAnyFieldInitialized = true;
    return false;
  }

  assert(V.getAs<loc::MemRegionVal>() &&
         "At this point V must be loc::MemRegionVal!");
  auto L = V.castAs<loc::MemRegionVal>();

  // We can't reason about symbolic regions, assume its initialized.
  // Note that this also avoids a potential infinite recursion, because
  // constructors for list-like classes are checked without being called, and
  // the Static Analyzer will construct a symbolic region for Node *next; or
  // similar code snippets.
  if (L.getRegion()->getSymbolicBase()) {
    IsAnyFieldInitialized = true;
    return false;
  }

  DynamicTypeInfo DynTInfo = getDynamicTypeInfo(State, L.getRegion());
  if (!DynTInfo.isValid()) {
    IsAnyFieldInitialized = true;
    return false;
  }

  QualType DynT = DynTInfo.getType();

  if (isVoidPointer(DynT)) {
    IsAnyFieldInitialized = true;
    return false;
  }

  // At this point the pointer itself is initialized and points to a valid
  // location, we'll now check the pointee.
  SVal DerefdV = State->getSVal(V.castAs<Loc>(), DynT);

  // If DerefdV is still a pointer value, we'll dereference it again (e.g.:
  // int** -> int*).
  while (auto Tmp = DerefdV.getAs<loc::MemRegionVal>()) {
    if (Tmp->getRegion()->getSymbolicBase()) {
      IsAnyFieldInitialized = true;
      return false;
    }

    DynTInfo = getDynamicTypeInfo(State, Tmp->getRegion());
    if (!DynTInfo.isValid()) {
      IsAnyFieldInitialized = true;
      return false;
    }

    DynT = DynTInfo.getType();
    if (isVoidPointer(DynT)) {
      IsAnyFieldInitialized = true;
      return false;
    }

    DerefdV = State->getSVal(*Tmp, DynT);
  }

  // If FR is a pointer pointing to a non-primitive type.
  if (Optional<nonloc::LazyCompoundVal> RecordV =
          DerefdV.getAs<nonloc::LazyCompoundVal>()) {

    const TypedValueRegion *R = RecordV->getRegion();

    if (DynT->getPointeeType()->isStructureOrClassType())
      return isNonUnionUninit(R, LocalChain.add(LocField(FR)));

    if (DynT->getPointeeType()->isUnionType()) {
      if (isUnionUninit(R)) {
        return addFieldToUninits(LocalChain.add(LocField(FR)));
      } else {
        IsAnyFieldInitialized = true;
        return false;
      }
    }

    if (DynT->getPointeeType()->isArrayType()) {
      IsAnyFieldInitialized = true;
      return false;
    }

    llvm_unreachable("All cases are handled!");
  }

  assert((isPrimitiveType(DynT->getPointeeType()) || DynT->isPointerType() ||
          DynT->isReferenceType()) &&
         "At this point FR must either have a primitive dynamic type, or it "
         "must be a null, undefined, unknown or concrete pointer!");

  if (isPrimitiveUninit(DerefdV))
    return addFieldToUninits(LocalChain.add(LocField(FR)));

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
