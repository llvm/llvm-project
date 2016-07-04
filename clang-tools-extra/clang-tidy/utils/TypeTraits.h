//===--- TypeTraits.h - clang-tidy-------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

/// Returns true if `Type` has a non-trivial move constructor.
bool hasNonTrivialMoveConstructor(QualType Type);

/// Return true if `Type` has a non-trivial move assignment operator.
bool hasNonTrivialMoveAssignment(QualType Type);

} // type_traits
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_TYPETRAITS_H
