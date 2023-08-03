//===--- TypeUtils.h - Type helper functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_REFACTOR_TYPE_UTILS_H
#define LLVM_CLANG_LIB_TOOLING_REFACTOR_TYPE_UTILS_H

#include "clang/AST/Type.h"

namespace clang {

class Decl;

namespace tooling {

/// \brief Find the most lexically appropriate type that can be used to describe
/// the return type of the given expression \p E.
///
/// When extracting code, we want to produce a function that returns a type
/// that matches the user's intent. This function can be used to find such a
/// type.
QualType findExpressionLexicalType(const Decl *FunctionLikeParentDecl,
                                   const Expr *E, QualType T,
                                   const PrintingPolicy &Policy,
                                   const ASTContext &Ctx);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_LIB_TOOLING_REFACTOR_TYPE_UTILS_H
