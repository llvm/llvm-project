// RUN: %clang_cc1 -O3 -ffp-contract=fast -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

float fp_contract_1(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_1fff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  return a * b + c;
}

float fp_contract_2(float a, float b, float c) {
  // CHECK-LABEL: fp_contract_2fff(
  // CHECK: fmul contract float
  // CHECK: fsub contract float
  return a * b - c;
}

void fp_contract_3(float *a, float b, float c) {
  // CHECK-LABEL: fp_contract_3Pfff(
  // CHECK: fmul contract float
  // CHECK: fadd contract float
  a[0] += b * c;
}

void fp_contract_4(float *a, float b, float c) {
  // CHECK-LABEL: fp_contract_4Pfff(
  // CHECK: fmul contract float
  // CHECK: fsub contract float
  a[0] -= b * c;
}
