// RUN: %clang_cc1  %s -w -emit-llvm -o -

float test(int X, ...) {
  __builtin_va_list ap;
  float F;
  __builtin_va_start(ap, X);
  F = __builtin_va_arg(ap, float);
  return F;
}
