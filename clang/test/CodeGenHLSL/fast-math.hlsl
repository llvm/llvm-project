// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=FAST
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fno-fast-math -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NO_FAST
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ffast-math -fno-fast-math -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NO_FAST
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fno-fast-math -ffast-math -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=FAST
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-library -fno-fast-math -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NO_FAST
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ffp-contract=fast -fno-fast-math -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CONTRACT
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fno-fast-math -ffp-contract=fast -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CONTRACT
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ffp-contract=off -fno-fast-math -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NO_CONTRACT
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fno-fast-math -ffp-contract=off -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NO_CONTRACT

// FAST-LABEL: define {{.*}}multiply_add{{.*}}(
// FAST: fmul fast float
// FAST: fadd fast float

// NO_FAST-LABEL: define {{.*}}multiply_add{{.*}}(
// NO_FAST: fmul float
// NO_FAST: fadd float

// CONTRACT-LABEL: define {{.*}}multiply_add{{.*}}(
// CONTRACT: fmul contract float
// CONTRACT: fadd contract float

// NO_CONTRACT-LABEL: define {{.*}}multiply_add{{.*}}(
// NO_CONTRACT: fmul float
// NO_CONTRACT: fadd float

float multiply_add(float a, float b, float c) {
  float product = a * b;
  return product + c;
}

// FAST-LABEL: define {{.*}}multiply_add_contract_on{{.*}}(
// FAST: fmul fast float
// FAST: fadd fast float

// NO_FAST-LABEL: define {{.*}}multiply_add_contract_on{{.*}}(
// NO_FAST: call float @llvm.fmuladd.f32(

// CONTRACT-LABEL: define {{.*}}multiply_add_contract_on{{.*}}(
// CONTRACT: fmul contract float
// CONTRACT: fadd contract float

// NO_CONTRACT-LABEL: define {{.*}}multiply_add_contract_on{{.*}}(
// NO_CONTRACT: fmul float
// NO_CONTRACT: fadd float

float multiply_add_contract_on(float a, float b, float c) {
  return a * b + c;
}
