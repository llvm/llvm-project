//===--- DeclRefExprUtils.h - clang-tidy-------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

namespace clang {
namespace tidy {
namespace decl_ref_expr_utils {

/// \brief Returns true if all DeclRefExpr to the variable within Stmt do not
/// modify it.
///
/// Returns true if only const methods or operators are called on the variable
/// or the variable is a const reference or value argument to a callExpr().
bool isOnlyUsedAsConst(const VarDecl &Var, const Stmt &Stmt,
                       ASTContext &Context);

} // namespace decl_ref_expr_utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_DECLREFEXPRUTILS_H
