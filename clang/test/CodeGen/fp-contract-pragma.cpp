// RUN: %clang_cc1 -O3 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

// Is FP_CONTRACT honored in a simple case?
float fp_contract_1(float a, float b, float c) {
// CHECK: _Z13fp_contract_1fff
// CHECK: tail call float @llvm.fmuladd
  #pragma STDC FP_CONTRACT ON
  return a * b + c;
}

// Is FP_CONTRACT state cleared on exiting compound statements?
float fp_contract_2(float a, float b, float c) {
// CHECK: _Z13fp_contract_2fff
// CHECK: %[[M:.+]] = fmul float %a, %b
// CHECK-NEXT: fadd float %[[M]], %c
  {
    #pragma STDC FP_CONTRACT ON
  }
  return a * b + c;  
}

// Does FP_CONTRACT survive template instantiation?
class Foo {};
Foo operator+(Foo, Foo);

template <typename T>
T template_muladd(T a, T b, T c) {
  #pragma STDC FP_CONTRACT ON
  return a * b + c;
}

float fp_contract_3(float a, float b, float c) {
// CHECK: _Z13fp_contract_3fff
// CHECK: tail call float @llvm.fmuladd
  return template_muladd<float>(a, b, c);
}

template<typename T> class fp_contract_4 {
  float method(float a, float b, float c) {
    #pragma STDC FP_CONTRACT ON
    return a * b + c;
  }
};

template class fp_contract_4<int>;
// CHECK: _ZN13fp_contract_4IiE6methodEfff
// CHECK: tail call float @llvm.fmuladd

// Check file-scoped FP_CONTRACT
#pragma STDC FP_CONTRACT ON
float fp_contract_5(float a, float b, float c) {
// CHECK: _Z13fp_contract_5fff
// CHECK: tail call float @llvm.fmuladd
  return a * b + c;
}

#pragma STDC FP_CONTRACT OFF
float fp_contract_6(float a, float b, float c) {
// CHECK: _Z13fp_contract_6fff
// CHECK: %[[M:.+]] = fmul float %a, %b
// CHECK-NEXT: fadd float %[[M]], %c
  return a * b + c;
}

// If the multiply has multiple uses, don't produce fmuladd.
// This used to assert (PR25719):
// https://llvm.org/bugs/show_bug.cgi?id=25719

float fp_contract_7(float a, float b, float c) {
// CHECK: _Z13fp_contract_7fff
// CHECK:  %[[M:.+]] = fmul float %b, 2.000000e+00
// CHECK-NEXT: fsub float %[[M]], %c
  #pragma STDC FP_CONTRACT ON
  return (a = 2 * b) - c;
}

float fp_contract_8(float a, float b, float c) {
// CHECK: _Z13fp_contract_8fff
// CHECK: fneg float %c
// CHECK: tail call float @llvm.fmuladd
  #pragma STDC FP_CONTRACT ON
  return a * b - c;
}

float fp_contract_9(float a, float b, float c) {
// CHECK: _Z13fp_contract_9fff
// CHECK: fneg float %a
// CHECK: tail call float @llvm.fmuladd
  #pragma STDC FP_CONTRACT ON
  return c - a * b;
}
