// RUN: %clang_cc1  -std=c99 %s -emit-llvm -o - | FileCheck %s
// CHECK: @Test
// CHECK-NOT: call{{ }}

_Bool A, B, C, D, E, F;
void TestF(float X, float Y) {
  A = __builtin_isgreater(X, Y);
  B = __builtin_isgreaterequal(X, Y);
  C = __builtin_isless(X, Y);
  D = __builtin_islessequal(X, Y);
  E = __builtin_islessgreater(X, Y);
  F = __builtin_isunordered(X, Y);
}
void TestD(double X, double Y) {
  A = __builtin_isgreater(X, Y);
  B = __builtin_isgreaterequal(X, Y);
  C = __builtin_isless(X, Y);
  D = __builtin_islessequal(X, Y);
  E = __builtin_islessgreater(X, Y);
  F = __builtin_isunordered(X, Y);
}
