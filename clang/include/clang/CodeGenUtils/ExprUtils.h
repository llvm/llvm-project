//===--- ExprUtils.h - Shared expression emission queries -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST queries about expressions that both classic CodeGen
// and CIR CodeGen need while emitting scalar, aggregate and lvalue
// expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_EXPRUTILS_H
#define LLVM_CLANG_CODEGENUTILS_EXPRUTILS_H

#include "clang/AST/ASTContext.h"

namespace clang::CodeGenUtils {

/// Strip off the variably-modified array types wrapping \p VLA and return the
/// first element type that has a fixed size.
QualType getFixedSizeElementType(const ASTContext &Ctx,
                                 const VariableArrayType *VLA);

/// Check whether the value of \p E is possibly a reference to or into a
/// __block variable.
bool isBlockVarRef(const Expr *E);

/// Check whether \p E is cheap enough and side-effect-free enough to evaluate
/// unconditionally instead of conditionally.  This is used to convert control
/// flow into selects in some cases.
bool isCheapEnoughToEvaluateUnconditionally(const Expr *E,
                                            const ASTContext &Ctx);

/// Check whether \p E is a trivial array filler, that is, one that is
/// equivalent to zero-initialization.
bool isTrivialFiller(const Expr *E);

/// Detect the unusual situation where an inline version of a builtin is
/// shadowed by a non-inline version.  In that case we should pick the external
/// one everywhere.  That's GCC behavior too.
bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *FD);

/// Given an object of the canonical type \p Type, can we safely copy a value
/// out of it based on its initializer?
///
/// FIXME: Overly conservative: rejects all arrays, and requires C++ records
/// to be trivial rather than just trivially copyable.
bool isConstantEmittableObjectType(QualType Type);

/// How a reference to a variable can be emitted as a constant.
enum ConstantEmissionKind {
  CEK_None,
  CEK_AsReferenceOnly,
  CEK_AsValueOrReference,
  CEK_AsValueOnly
};

/// Can we constant-emit a load of a reference to a variable of the given
/// type?  This is different from predicates like
/// Decl::mightBeUsableInConstantExpressions because we do want it to apply
/// in situations that don't necessarily satisfy the language's rules
/// for this (e.g. C++'s ODR-use rules).  For example, we want to be able
/// to do this with const float variables even if those variables
/// aren't marked 'constexpr'.
///
/// FIXME: For references, checks the pointee type, not the bound object's.
ConstantEmissionKind checkVarTypeForConstantEmission(QualType Type);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_EXPRUTILS_H
