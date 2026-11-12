// RUN: %clang_cc1 -emit-llvm-only -verify %s
// expected-no-diagnostics

// Make sure we don't crash generating y; its value is constant, but the
// initializer has side effects, so EmitConstantExpr should fail.
int x();
int y = x() & 0;
