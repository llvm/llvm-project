// RUN: %clang_cc1 -O3 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -O3 -triple %itanium_abi_triple -freciprocal-math -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,FLAG %s

float base(float a, float b, float c) {
// CHECK-LABEL: _Z4basefff
// FLAG: %[[A:.+]] = fdiv arcp float %b, %c
// FLAG: %[[M:.+]] = fdiv arcp float %[[A]], %b
// FLAG-NEXT: fadd arcp float %[[M]], %c

// DEFAULT: %[[A:.+]] = fdiv float %b, %c
// DEFAULT: %[[M:.+]] = fdiv float %[[A]], %b
// DEFAULT-NEXT: fadd float %[[M]], %c
  a = b / c;
  return a / b + c;
}

// Simple case
float fp_recip_simple(float a, float b, float c) {
// CHECK-LABEL: _Z15fp_recip_simplefff
// CHECK: %[[A:.+]] = fdiv arcp float %b, %c
// CHECK: %[[M:.+]] = fdiv arcp float %[[A]], %b
// CHECK-NEXT: fadd arcp float %[[M]], %c
#pragma clang fp reciprocal(on)
  a = b / c;
  return a / b + c;
}

// Test interaction with -freciprocal-math
float fp_recip_disable(float a, float b, float c) {
// CHECK-LABEL: _Z16fp_recip_disablefff
// CHECK: %[[A:.+]] = fdiv float %b, %c
// CHECK: %[[M:.+]] = fdiv float %[[A]], %b
// CHECK-NEXT: fadd float %[[M]], %c
#pragma clang fp reciprocal(off)
  a = b / c;
  return a / b + c;
}

float fp_recip_with_reassoc_simple(float a, float b, float c) {
// CHECK-LABEL: _Z28fp_recip_with_reassoc_simplefff
// CHECK: %[[A:.+]] = fmul reassoc arcp float %b, %c
// CHECK: %[[M:.+]] = fdiv reassoc arcp float %b, %[[A]]
// CHECK-NEXT: fadd reassoc arcp float %[[M]], %c
#pragma clang fp reciprocal(on) reassociate(on)
  a = b / c;
  return a / b + c;
}

// arcp pragma should only apply to its scope
float fp_recip_scoped(float a, float b, float c) {
  // CHECK-LABEL: _Z15fp_recip_scopedfff
  // DEFAULT: %[[M:.+]] = fdiv float %a, %b
  // DEFAULT-NEXT: fadd float %[[M]], %c
  // FLAG: %[[M:.+]] = fdiv arcp float %a, %b
  // FLAG-NEXT: fadd arcp float %[[M]], %c
  {
#pragma clang fp reciprocal(on)
  }
  return a / b + c;
}

// arcp pragma should apply to templates as well
class Foo {};
Foo operator+(Foo, Foo);
template <typename T>
T template_recip(T a, T b, T c) {
#pragma clang fp reciprocal(on)
  return ((a / b) - c) + c;
}

float fp_recip_template(float a, float b, float c) {
  // CHECK-LABEL: _Z17fp_recip_templatefff
  // CHECK: %[[A1:.+]] = fdiv arcp float %a, %b
  // CHECK-NEXT: %[[A2:.+]] = fsub arcp float %[[A1]], %c
  // CHECK-NEXT: fadd arcp float %[[A2]], %c
  return template_recip<float>(a, b, c);
}

// File Scoping should work across functions
#pragma clang fp reciprocal(on)
float fp_file_scope_on(float a, float b, float c) {
  // CHECK-LABEL: _Z16fp_file_scope_onfff
  // CHECK: %[[M1:.+]] = fdiv arcp float %a, %c
  // CHECK-NEXT: %[[M2:.+]] = fdiv arcp float %b, %c
  // CHECK-NEXT: fadd arcp float %[[M1]], %[[M2]]
  return (a / c) + (b / c);
}

// Inner pragma has precedence
float fp_file_scope_stop(float a, float b, float c) {
  // CHECK-LABEL: _Z18fp_file_scope_stopfff
  // CHECK: %[[A:.+]] = fdiv arcp float %a, %a
  // CHECK: %[[M1:.+]] = fdiv float %[[A]], %c
  // CHECK-NEXT: %[[M2:.+]] = fdiv float %b, %c
  // CHECK-NEXT: fsub float %[[M1]], %[[M2]]
  a = a / a;
  {
#pragma clang fp reciprocal(off)
    return (a / c) - (b / c);
  }
}

#pragma clang fp reciprocal(off)
float fp_recip_off(float a, float b, float c) {
  // CHECK-LABEL: _Z12fp_recip_offfff
  // CHECK: %[[D1:.+]] = fdiv float %a, %c
  // CHECK-NEXT: %[[D2:.+]] = fdiv float %b, %c
  // CHECK-NEXT: fadd float %[[D1]], %[[D2]]
  return (a / c) + (b / c);
}

// Takes latest flag
float fp_recip_many(float a, float b, float c) {
// CHECK-LABEL: _Z13fp_recip_manyfff
// CHECK: %[[D1:.+]] = fdiv arcp float %a, %c
// CHECK-NEXT: %[[D2:.+]] = fdiv arcp float %b, %c
// CHECK-NEXT: fadd arcp float %[[D1]], %[[D2]]
#pragma clang fp reciprocal(off) reciprocal(on)
  return (a / c) + (b / c);
}

// Pragma does not propagate through called functions
float helper_func(float a, float b, float c) { return a + b + c; }
float fp_recip_call_helper(float a, float b, float c) {
// CHECK-LABEL: _Z20fp_recip_call_helperfff
// CHECK: %[[S1:.+]] = fadd float %a, %b
// CHECK-NEXT: fadd float %[[S1]], %c
#pragma clang fp reciprocal(on)
  return helper_func(a, b, c);
}
