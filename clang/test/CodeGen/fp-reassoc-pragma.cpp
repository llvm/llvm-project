// RUN: %clang_cc1 -O3 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s
// Simple case
float fp_reassoc_simple(float a, float b, float c) {
// CHECK: _Z17fp_reassoc_simplefff
// CHECK: %[[A:.+]] = fadd reassoc float %b, %c
// CHECK: %[[M:.+]] = fmul reassoc float %[[A]], %b
// CHECK-NEXT: fadd reassoc float %[[M]], %c
#pragma clang fp reassociate(on)
  a = b + c;
  return a * b + c;
}

// Reassoc pragma should only apply to its scope
float fp_reassoc_scoped(float a, float b, float c) {
  // CHECK: _Z17fp_reassoc_scopedfff
  // CHECK: %[[M:.+]] = fmul float %a, %b
  // CHECK-NEXT: fadd float %[[M]], %c
  {
#pragma clang fp reassociate(on)
  }
  return a * b + c;
}

// Reassoc pragma should apply to templates as well
class Foo {};
Foo operator+(Foo, Foo);
template <typename T>
T template_reassoc(T a, T b, T c) {
#pragma clang fp reassociate(on)
  return ((a + b) - c) + c;
}

float fp_reassoc_template(float a, float b, float c) {
  // CHECK: _Z19fp_reassoc_templatefff
  // CHECK: %[[A1:.+]] = fadd reassoc float %a, %b
  // CHECK-NEXT: %[[A2:.+]] = fsub reassoc float %[[A1]], %c
  // CHECK-NEXT: fadd reassoc float %[[A2]], %c
  return template_reassoc<float>(a, b, c);
}

// File Scoping should work across functions
#pragma clang fp reassociate(on)
float fp_file_scope_on(float a, float b, float c) {
  // CHECK: _Z16fp_file_scope_onfff
  // CHECK: %[[M1:.+]] = fmul reassoc float %a, %c
  // CHECK-NEXT: %[[M2:.+]] = fmul reassoc float %b, %c
  // CHECK-NEXT: fadd reassoc float %[[M1]], %[[M2]]
  return (a * c) + (b * c);
}

// Inner pragma has precedence
float fp_file_scope_stop(float a, float b, float c) {
  // CHECK: _Z18fp_file_scope_stopfff
  // CHECK: %[[A:.+]] = fadd reassoc float %a, %a
  // CHECK: %[[M1:.+]] = fmul float %[[A]], %c
  // CHECK-NEXT: %[[M2:.+]] = fmul float %b, %c
  // CHECK-NEXT: fsub float %[[M1]], %[[M2]]
  a = a + a;
  {
#pragma clang fp reassociate(off)
    return (a * c) - (b * c);
  }
}

#pragma clang fp reassociate(off)
float fp_reassoc_off(float a, float b, float c) {
  // CHECK: _Z14fp_reassoc_offfff
  // CHECK: %[[D1:.+]] = fdiv float %a, %c
  // CHECK-NEXT: %[[D2:.+]] = fdiv float %b, %c
  // CHECK-NEXT: fadd float %[[D1]], %[[D2]]
  return (a / c) + (b / c);
}

// Takes latest flag
float fp_reassoc_many(float a, float b, float c) {
// CHECK: _Z15fp_reassoc_manyfff
// CHECK: %[[D1:.+]] = fdiv reassoc float %a, %c
// CHECK-NEXT: %[[D2:.+]] = fdiv reassoc float %b, %c
// CHECK-NEXT: fadd reassoc float %[[D1]], %[[D2]]
#pragma clang fp reassociate(off) reassociate(on)
  return (a / c) + (b / c);
}

// Pragma does not propagate through called functions
float helper_func(float a, float b, float c) { return a + b + c; }
float fp_reassoc_call_helper(float a, float b, float c) {
// CHECK: _Z22fp_reassoc_call_helperfff
// CHECK: %[[S1:.+]] = fadd float %a, %b
// CHECK-NEXT: fadd float %[[S1]], %c
#pragma clang fp reassociate(on)
  return helper_func(a, b, c);
}
