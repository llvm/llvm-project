// RUN: %clang_cc1 -triple aarch64-arm-none-eabi -target-feature +bf16 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK64
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi hard -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK32-HARD
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi softfp -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK32-SOFTFP

// CHECK64: define {{.*}}void @_Z3foou6__bf16(bfloat %b)
// CHECK32-HARD: define {{.*}}void @_Z3foou6__bf16(bfloat %b)
// CHECK32-SOFTFP: define {{.*}}void @_Z3foou6__bf16(i32 %b.coerce)
void foo(__bf16 b) {}
