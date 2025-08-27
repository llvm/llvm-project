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
