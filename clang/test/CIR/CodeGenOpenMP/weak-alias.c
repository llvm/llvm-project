// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR --implicit-check-not=One --implicit-check-not=Missing
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM --implicit-check-not=One --implicit-check-not=Missing
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG --implicit-check-not=One --implicit-check-not=Missing

// Aliasee without declare target: skipped on the device.
int __One(void) { return 1; }
int One(void) __attribute__((weak, alias("__One")));
int One_(void) __attribute__((alias("__One")));

// Declare target aliasee: emitted on the device.
// CIR-DAG: cir.func weak hidden @Two() -> !s32i alias(@__Two)
// CIR-DAG: cir.func hidden dso_local @Two_() -> !s32i alias(@__Two)
// LLVM-DAG: @Two = weak hidden alias i32 (), ptr @__Two
// LLVM-DAG: @Two_ = hidden alias i32 (), ptr @__Two
// OGCG-DAG: @Two = weak hidden alias i32 (), ptr @__Two
// OGCG-DAG: @Two_ = hidden alias i32 (), ptr @__Two
#pragma omp declare target
int __Two(void) { return 2; }
int Two(void) __attribute__((weak, alias("__Two")));
int Two_(void) __attribute__((alias("__Two")));
#pragma omp end declare target

// Missing aliasee: skipped on the device without a diagnostic.
int Missing(void) __attribute__((alias("__Missing")));
