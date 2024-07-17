// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -o - -emit-llvm -fmath-errno %s | FileCheck %s

float _Complex test__builtin_conjf(float _Complex x) {
// CHECK-LABEL: @test__builtin_conjf(
// CHECK: fneg float %x.imag
  return __builtin_conjf(x);
}

double _Complex test__builtin_conj(double _Complex x) {
// CHECK-LABEL: @test__builtin_conj(
// CHECK: fneg double %x.imag
  return __builtin_conj(x);
}

long double _Complex test__builtin_conjl(long double _Complex x) {
// CHECK-LABEL: @test__builtin_conjl(
// CHECK: fneg x86_fp80 %x.imag
  return __builtin_conjl(x);
}
