// RUN: %clang_cc1 -fsyntax-only -verify %s

// Regression test for the "function-like macro used without parens"
// diagnostic added in https://github.com/llvm/llvm-project/pull/123495.
// Confirms the same diagnostic fires correctly in C++ (not just C),
// since the implementation lives in the shared SemaExpr.cpp.

#define BAZ() 1
// expected-note@-1 {{'BAZ' defined here as a function-like macro}}

int x = BAZ; // expected-error {{'BAZ' is defined as a function-like macro; did you mean 'BAZ(...)'?}}