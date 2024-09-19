// RUN: %clang_cc1 -O3 -ffp-contract=fast-honor-pragmas -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

float fp_contract_on_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_on_1fff(
  // CHECK: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})
  #pragma STDC FP_CONTRACT ON
  return a * b + c;
}

float fp_contract_on_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_on_2fff(
  // CHECK: fmul float
  // CHECK: fadd float
  #pragma STDC FP_CONTRACT ON
  float t = a * b;
  return t + c;
}

float fp_contract_off_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_off_1fff(
  // CHECK: fmul float
  // CHECK: fadd float
  #pragma STDC FP_CONTRACT OFF
  return a * b + c;
}

float fp_contract_off_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_off_2fff(
  // CHECK: fmul float
  // CHECK: fadd float
  #pragma STDC FP_CONTRACT OFF
  float t = a * b;
  return t + c;
}

float fp_contract_default_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_default_1fff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  #pragma STDC FP_CONTRACT DEFAULT
  return a * b + c;
}

float fp_contract_default_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_default_2fff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  #pragma STDC FP_CONTRACT DEFAULT
  float t = a * b;
  return t + c;
}

float fp_contract_clang_on_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_clang_on_1fff(
  // CHECK: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})
  #pragma clang fp contract(on)
  return a * b + c;
}

float fp_contract_clang_on_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_clang_on_2fff(
  // CHECK: fmul float
  // CHECK: fadd float
  #pragma clang fp contract(on)
  float t = a * b;
  return t + c;
}

float fp_contract_clang_off_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_clang_off_1fff(
  // CHECK: fmul float
  // CHECK: fadd float
  #pragma clang fp contract(off)
  return a * b + c;
}

float fp_contract_clang_off_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_clang_off_2fff(
  // CHECK: fmul float
  // CHECK: fadd float
  #pragma clang fp contract(off)
  float t = a * b;
  return t + c;
}

float fp_contract_clang_fast_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_clang_fast_1fff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  #pragma clang fp contract(fast)
  return a * b + c;
}

float fp_contract_clang_fast_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_clang_fast_2fff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  #pragma clang fp contract(fast)
  float t = a * b;
  return t + c;
}

#pragma STDC FP_CONTRACT ON

float fp_contract_global_on_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_global_on_1fff(
  // CHECK: call float @llvm.fmuladd.f32(float {{.*}}, float {{.*}}, float {{.*}})
  return a * b + c;
}

float fp_contract_global_on_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_global_on_2fff(
  // CHECK: fmul float
  // CHECK: fadd float
  float t = a * b;
  return t + c;
}

#pragma STDC FP_CONTRACT OFF

float fp_contract_global_off_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_global_off_1fff(
  // CHECK: fmul float
  // CHECK: fadd float
  return a * b + c;
}

float fp_contract_global_off_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_global_off_2fff(
  // CHECK: fmul float
  // CHECK: fadd float
  float t = a * b;
  return t + c;
}

#pragma STDC FP_CONTRACT DEFAULT

float fp_contract_global_default_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_global_default_1fff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  return a * b + c;
}

float fp_contract_global_default_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_global_default_2fff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  float t = a * b;
  return t + c;
}
