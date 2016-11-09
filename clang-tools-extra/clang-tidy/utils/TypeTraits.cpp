//===--- TypeTraits.cpp - clang-tidy---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypeTraits.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace tidy {
namespace utils {
namespace type_traits {

namespace {

bool classHasTrivialCopyAndDestroy(QualType Type) {
  auto *Record = Type->getAsCXXRecordDecl();
  return Record && Record->hasDefinition() &&
         !Record->hasNonTrivialCopyConstructor() &&
         !Record->hasNonTrivialDestructor();
}

bool hasDeletedCopyConstructor(QualType Type) {
  auto *Record = Type->getAsCXXRecordDecl();
  if (!Record || !Record->hasDefinition())
    return false;
  for (const auto *Constructor : Record->ctors()) {
    if (Constructor->isCopyConstructor() && Constructor->isDeleted())
      return true;
  }
  return false;
}

} // namespace

llvm::Optional<bool> isExpensiveToCopy(QualType Type,
                                       const ASTContext &Context) {
  if (Type->isDependentType() || Type->isIncompleteType())
    return llvm::None;
  return !Type.isTriviallyCopyableType(Context) &&
         !classHasTrivialCopyAndDestroy(Type) &&
         !hasDeletedCopyConstructor(Type);
}

bool recordIsTriviallyDefaultConstructible(const RecordDecl &RecordDecl,
                                           const ASTContext &Context) {
  const auto *ClassDecl = dyn_cast<CXXRecordDecl>(&RecordDecl);
  // Non-C++ records are always trivially constructible.
  if (!ClassDecl)
    return true;
  // A class with a user-provided default constructor is not trivially
  // constructible.
  if (ClassDecl->hasUserProvidedDefaultConstructor())
    return false;
  // A polymorphic class is not trivially constructible
  if (ClassDecl->isPolymorphic())
    return false;
  // A class is trivially constructible if it has a trivial default constructor.
  if (ClassDecl->hasTrivialDefaultConstructor())
    return true;

  // If all its fields are trivially constructible and have no default
  // initializers.
  for (const FieldDecl *Field : ClassDecl->fields()) {
    if (Field->hasInClassInitializer())
      return false;
    if (!isTriviallyDefaultConstructible(Field->getType(), Context))
      return false;
  }
  // If all its direct bases are trivially constructible.
  for (const CXXBaseSpecifier &Base : ClassDecl->bases()) {
    if (!isTriviallyDefaultConstructible(Base.getType(), Context))
      return false;
    if (Base.isVirtual())
      return false;
  }

  return true;
}

// Based on QualType::isTrivial.
bool isTriviallyDefaultConstructible(QualType Type, const ASTContext &Context) {
  if (Type.isNull())
    return false;

  if (Type->isArrayType())
    return isTriviallyDefaultConstructible(Context.getBaseElementType(Type),
                                           Context);

  // Return false for incomplete types after skipping any incomplete array
  // types which are expressly allowed by the standard and thus our API.
  if (Type->isIncompleteType())
    return false;

  if (Context.getLangOpts().ObjCAutoRefCount) {
    switch (Type.getObjCLifetime()) {
    case Qualifiers::OCL_ExplicitNone:
      return true;

    case Qualifiers::OCL_Strong:
    case Qualifiers::OCL_Weak:
    case Qualifiers::OCL_Autoreleasing:
      return false;

    case Qualifiers::OCL_None:
      if (Type->isObjCLifetimeType())
        return false;
      break;
    }
  }

  QualType CanonicalType = Type.getCanonicalType();
  if (CanonicalType->isDependentType())
    return false;

  // As an extension, Clang treats vector types as Scalar types.
  if (CanonicalType->isScalarType() || CanonicalType->isVectorType())
    return true;

  if (const auto *RT = CanonicalType->getAs<RecordType>()) {
    return recordIsTriviallyDefaultConstructible(*RT->getDecl(), Context);
  }

  // No other types can match.
  return false;
}

bool hasNonTrivialMoveConstructor(QualType Type) {
  auto *Record = Type->getAsCXXRecordDecl();
  return Record && Record->hasDefinition() &&
         Record->hasNonTrivialMoveConstructor();
}

bool hasNonTrivialMoveAssignment(QualType Type) {
  auto *Record = Type->getAsCXXRecordDecl();
  return Record && Record->hasDefinition() &&
         Record->hasNonTrivialMoveAssignment();
}

} // namespace type_traits
} // namespace utils
} // namespace tidy
} // namespace clang
