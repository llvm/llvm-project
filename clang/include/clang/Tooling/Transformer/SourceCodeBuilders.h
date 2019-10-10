//===--- SourceCodeBuilders.h - Source-code building facilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file collects facilities for generating source code strings.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TRANSFORMER_SOURCE_CODE_BUILDERS_H_
#define LLVM_CLANG_TOOLING_TRANSFORMER_SOURCE_CODE_BUILDERS_H_

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include <string>

namespace clang {
namespace tooling {

/// \name Code analysis utilities.
/// @{
/// Ignores implicit object-construction expressions in addition to the normal
/// implicit expressions that are ignored.
const Expr *reallyIgnoreImplicit(const Expr &E);

/// Determines whether printing this expression in *any* expression requires
/// parentheses to preserve its meaning. This analyses is necessarily
/// conservative because it lacks information about the target context.
bool mayEverNeedParens(const Expr &E);

/// Determines whether printing this expression to the left of a dot or arrow
/// operator requires a parentheses to preserve its meaning. Given that
/// dot/arrow are (effectively) the highest precedence, this is equivalent to
/// asking whether it ever needs parens.
inline bool needParensBeforeDotOrArrow(const Expr &E) {
  return mayEverNeedParens(E);
}

/// Determines whether printing this expression to the right of a unary operator
/// requires a parentheses to preserve its meaning.
bool needParensAfterUnaryOperator(const Expr &E);
/// @}

/// \name Basic code-string generation utilities.
/// @{

/// Builds source for an expression, adding parens if needed for unambiguous
/// parsing.
llvm::Optional<std::string> buildParens(const Expr &E,
                                        const ASTContext &Context);

/// Builds idiomatic source for the dereferencing of `E`: prefix with `*` but
/// simplify when it already begins with `&`.  \returns empty string on failure.
llvm::Optional<std::string> buildDereference(const Expr &E,
                                             const ASTContext &Context);

/// Builds idiomatic source for taking the address of `E`: prefix with `&` but
/// simplify when it already begins with `*`.  \returns empty string on failure.
llvm::Optional<std::string> buildAddressOf(const Expr &E,
                                           const ASTContext &Context);

/// Adds a dot to the end of the given expression, but adds parentheses when
/// needed by the syntax, and simplifies to `->` when possible, e.g.:
///
///  `x` becomes `x.`
///  `*a` becomes `a->`
///  `a+b` becomes `(a+b).`
llvm::Optional<std::string> buildDot(const Expr &E, const ASTContext &Context);

/// Adds an arrow to the end of the given expression, but adds parentheses
/// when needed by the syntax, and simplifies to `.` when possible, e.g.:
///
///  `x` becomes `x->`
///  `&a` becomes `a.`
///  `a+b` becomes `(a+b)->`
llvm::Optional<std::string> buildArrow(const Expr &E,
                                       const ASTContext &Context);
/// @}

} // namespace tooling
} // namespace clang
#endif // LLVM_CLANG_TOOLING_TRANSFORMER_SOURCE_CODE_BUILDERS_H_
