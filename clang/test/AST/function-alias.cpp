// RUN: %clang_cc1 -std=c++17 -fsyntax-only %s

// Verify that ASTContext::getFunctionTypeWithExceptionSpec (called through
// ASTContext::hasSameFunctionTypeIgnoringExceptionSpec from
// ExprEvaluatorBase::handleCallExpr in lib/AST/ExprConstant.cpp) does not crash
// for a type alias.

constexpr int f() noexcept { return 0; }

using F = int();

constexpr int g(F * p) { return p(); }

constexpr int n = g(f);
