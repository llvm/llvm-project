//===--- TypeTraits.h - clang-tidy-------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_TYPETRAITS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_TYPETRAITS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

namespace clang {
namespace tidy {
namespace utils {
namespace type_traits {

/// Returns `true` if `Type` is expensive to copy.
llvm::Optional<bool> isExpensiveToCopy(QualType Type,
                                       const ASTContext &Context);

/// Returns `true` if `Type` is trivially default constructible.
bool isTriviallyDefaultConstructible(QualType Type, const ASTContext &Context);

/// Returns `true` if `RecordDecl` is trivially default constructible.
bool recordIsTriviallyDefaultConstructible(const RecordDecl &RecordDecl,
                                           const ASTContext &Context);

/// Returns `true` if `Type` is trivially destructible.
bool isTriviallyDestructible(QualType Type);

/// Returns true if `Type` has a non-trivial move constructor.
bool hasNonTrivialMoveConstructor(QualType Type);

/// Return true if `Type` has a non-trivial move assignment operator.
bool hasNonTrivialMoveAssignment(QualType Type);

} // type_traits
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_TYPETRAITS_H
