// REQUIRES: aarch64-registered-target
// RUN: not %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -emit-llvm -o - %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple arm64-apple-ios7 -target-abi darwinpcs -target-feature +sve -fallow-half-arguments-and-returns -emit-llvm -o - %s 2>&1 | FileCheck %s

// CHECK: Passing SVE types to variadic functions is currently not supported

#include <arm_sve.h>
#include <stdarg.h>

double foo(char *str, ...) {
  va_list ap;
  svfloat64_t v;
  double x;

  va_start(ap, str);
  v = va_arg(ap, svfloat64_t);
  x = va_arg(ap, double);
  va_end(ap);

  return x + svaddv(svptrue_b8(), v);
}
