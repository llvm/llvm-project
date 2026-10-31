//===--- CallUtils.h - Shared call emission queries -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST and language option queries that both classic
// CodeGen and CIR CodeGen need while building a call and its argument and
// return value attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_CALLUTILS_H
#define LLVM_CLANG_CODEGENUTILS_CALLUTILS_H

#include "clang/AST/ASTContext.h"
#include "llvm/ADT/FloatingPointMode.h"

namespace clang {
class CallExpr;
class TargetCXXABI;
} // namespace clang

namespace clang::CodeGenUtils {

/// The order in which the arguments of a call must be evaluated.
enum class EvaluationOrder {
  ///! No language constraints on evaluation order.
  Default,
  ///! Language semantics require left-to-right evaluation.
  ForceLeftToRight,
  ///! Language semantics require right-to-left evaluation.
  ForceRightToLeft
};

/// Returns the argument evaluation order the language requires for the given
/// call. C++17 requires that arguments to a call using assignment syntax be
/// evaluated right-to-left, and that arguments to certain other operators be
/// evaluated left-to-right.
EvaluationOrder getCallArgEvaluationOrder(const CallExpr *E);

/// Returns true if call arguments should be emitted left-to-right, given the
/// order the language requires and the C++ ABI's default. The MS C++ ABI
/// evaluates right-to-left because arguments are destroyed left-to-right in
/// the callee; a language-required order overrides that.
bool shouldEvaluateArgsLeftToRight(EvaluationOrder Order,
                                   const TargetCXXABI &ABI);

/// Returns the canonical formal type of the given C++ method.
CanQual<FunctionProtoType> getFormalType(const CXXMethodDecl *MD);

/// Returns the set of floating-point value kinds that the language options
/// promise never reach a function's arguments or return value.
llvm::FPClassTest getNoFPClassTestMask(const LangOptions &LangOpts);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_CALLUTILS_H
