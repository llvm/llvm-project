// RUN: %clang_cc1  %s -w -emit-llvm -o -

// https://bugs.llvm.org/show_bug.cgi?id=46644#c6
// XFAIL: arm64-apple

float test(int X, ...) {
  __builtin_va_list ap;
  float F;
  __builtin_va_start(ap, X);
  F = __builtin_va_arg(ap, float);
  return F;
}
