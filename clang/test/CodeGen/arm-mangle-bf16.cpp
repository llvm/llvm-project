// RUN: %clang_cc1 -triple aarch64 -target-feature +bf16 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-feature -bf16 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi hard -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi softfp -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +bf16 -emit-llvm -o - %s | llvm-cxxfilt -n | FileCheck %s --check-prefix=DEMANGLE
// RUN: %clang_cc1 -triple aarch64 -target-feature -bf16 -emit-llvm -o - %s | llvm-cxxfilt -n | FileCheck %s --check-prefix=DEMANGLE
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi hard -emit-llvm -o - %s | llvm-cxxfilt -n | FileCheck %s --check-prefix=DEMANGLE
// RUN: %clang_cc1 -triple arm-arm-none-eabi -target-feature +bf16 -mfloat-abi softfp -emit-llvm -o - %s | llvm-cxxfilt -n | FileCheck %s --check-prefix=DEMANGLE

// CHECK:    define {{.*}}void @_Z3foou6__bf16(bfloat noundef %b)
// DEMANGLE: define {{.*}}void @foo(__bf16)
void foo(__bf16 b) {}

struct bar;

// CHECK:    define {{.*}}void @_Z10substituteu6__bf16R3barS1_
// DEMANGLE: define {{.*}}void @substitute(__bf16, bar&, bar&)
void substitute(__bf16 a, bar &b, bar &c) {
}
