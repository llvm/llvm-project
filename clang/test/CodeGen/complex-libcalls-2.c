// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -S -o - -emit-llvm              %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -S -o - -emit-llvm -fmath-errno %s | FileCheck %s

_Complex float conjf(_Complex float);
_Complex double conj(_Complex double);
_Complex long double conjl(_Complex long double);

float _Complex test_conjf(float _Complex x) {
// CHECK-LABEL: @test_conjf(
// CHECK: fneg float %x.imag
  return conjf(x);
}

double _Complex test_conj(double _Complex x) {
// CHECK-LABEL: @test_conj(
// CHECK: fneg double %x.imag
  return conj(x);
}

long double _Complex test_conjl(long double _Complex x) {
// CHECK-LABEL: @test_conjl(
// CHECK: fneg x86_fp80 %x.imag
  return conjl(x);
}
