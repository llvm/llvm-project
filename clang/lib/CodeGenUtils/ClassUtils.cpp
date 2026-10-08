//===--- ClassUtils.cpp - Shared C++ class emission queries ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/ClassUtils.h"
#include "clang/AST/EvaluatedExprVisitor.h"

namespace clang::CodeGenUtils {
namespace {
/// A visitor which checks whether an initializer uses 'this' in a
/// way which requires the vtable to be properly set.
struct DynamicThisUseChecker
    : ConstEvaluatedExprVisitor<DynamicThisUseChecker> {
  using super = ConstEvaluatedExprVisitor<DynamicThisUseChecker>;

  bool UsesThis = false;

  DynamicThisUseChecker(const ASTContext &C) : super(C) {}

  // Black-list all explicit and implicit references to 'this'.
  //
  // Do we need to worry about external references to 'this' derived
  // from arbitrary code?  If so, then anything which runs arbitrary
  // external code might potentially access the vtable.
  void VisitCXXThisExpr(const CXXThisExpr *E) { UsesThis = true; }
};
} // namespace

bool baseInitializerUsesThis(ASTContext &Ctx, const Expr *Init) {
  DynamicThisUseChecker Checker(Ctx);
  Checker.Visit(Init);
  return Checker.UsesThis;
}

} // namespace clang::CodeGenUtils
