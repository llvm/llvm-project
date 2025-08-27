//===- Type.h - C Language Family Type Representation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// C Language Family Type Representation
///
/// This file defines some inline methods for clang::Type which depend on
/// Decl.h, avoiding a circular dependency.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPE_H
#define LLVM_CLANG_AST_TYPE_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"

namespace clang {

inline CXXRecordDecl *Type::getAsCXXRecordDecl() const {
  const auto *TT = dyn_cast<TagType>(CanonicalType);
  if (!isa_and_present<RecordType, InjectedClassNameType>(TT))
    return nullptr;
  auto *TD = TT->getOriginalDecl();
  if (isa<RecordType>(TT) && !isa<CXXRecordDecl>(TD))
    return nullptr;
  return cast<CXXRecordDecl>(TD)->getDefinitionOrSelf();
}

inline CXXRecordDecl *Type::castAsCXXRecordDecl() const {
  const auto *TT = cast<TagType>(CanonicalType);
  return cast<CXXRecordDecl>(TT->getOriginalDecl())->getDefinitionOrSelf();
}

inline RecordDecl *Type::getAsRecordDecl() const {
  const auto *TT = dyn_cast<TagType>(CanonicalType);
  if (!isa_and_present<RecordType, InjectedClassNameType>(TT))
    return nullptr;
  return cast<RecordDecl>(TT->getOriginalDecl())->getDefinitionOrSelf();
}

inline RecordDecl *Type::castAsRecordDecl() const {
  const auto *TT = cast<TagType>(CanonicalType);
  return cast<RecordDecl>(TT->getOriginalDecl())->getDefinitionOrSelf();
}

inline EnumDecl *Type::getAsEnumDecl() const {
  if (const auto *TT = dyn_cast<EnumType>(CanonicalType))
    return TT->getOriginalDecl()->getDefinitionOrSelf();
  return nullptr;
}

inline EnumDecl *Type::castAsEnumDecl() const {
  return cast<EnumType>(CanonicalType)
      ->getOriginalDecl()
      ->getDefinitionOrSelf();
}

inline TagDecl *Type::getAsTagDecl() const {
  if (const auto *TT = dyn_cast<TagType>(CanonicalType))
    return TT->getOriginalDecl()->getDefinitionOrSelf();
  return nullptr;
}

inline TagDecl *Type::castAsTagDecl() const {
  return cast<TagType>(CanonicalType)->getOriginalDecl()->getDefinitionOrSelf();
}

inline bool QualType::hasNonTrivialToPrimitiveDefaultInitializeCUnion() const {
  if (auto *RD = getTypePtr()->getBaseElementTypeUnsafe()->getAsRecordDecl())
    return hasNonTrivialToPrimitiveDefaultInitializeCUnion(RD);
  return false;
}

inline bool QualType::hasNonTrivialToPrimitiveDestructCUnion() const {
  if (auto *RD = getTypePtr()->getBaseElementTypeUnsafe()->getAsRecordDecl())
    return hasNonTrivialToPrimitiveDestructCUnion(RD);
  return false;
}

inline bool QualType::hasNonTrivialToPrimitiveCopyCUnion() const {
  if (auto *RD = getTypePtr()->getBaseElementTypeUnsafe()->getAsRecordDecl())
    return hasNonTrivialToPrimitiveCopyCUnion(RD);
  return false;
}

} // namespace clang

#endif // LLVM_CLANG_AST_TYPE_H
