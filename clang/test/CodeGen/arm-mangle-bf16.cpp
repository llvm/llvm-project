// RUN: %clang_cc1 -triple aarch64-arm-none-eabi -target-feature +bf16 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi hard -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi softfp -emit-llvm -o - %s | FileCheck %s

// CHECK: define {{.*}}void @_Z3foou6__bf16(bfloat noundef %b)
void foo(__bf16 b) {}
