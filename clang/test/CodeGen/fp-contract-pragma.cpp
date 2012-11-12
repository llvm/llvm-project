// RUN: %clang_cc1 -O3 -emit-llvm -o - %s | FileCheck %s

// Is FP_CONTRACT is honored in a simple case?
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

// Does FP_CONTRACT survive template instatiation?
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

