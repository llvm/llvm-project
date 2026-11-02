// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x hip -triple x86_64-unknown-linux-gnu -aux-triple amdgpu-amd-amdhsa \
// RUN:   -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST --implicit-check-not=One --implicit-check-not=Missing
// RUN: %clang_cc1 -x hip -triple x86_64-unknown-linux-gnu -aux-triple amdgpu-amd-amdhsa \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM-HOST --implicit-check-not=One --implicit-check-not=Missing
// RUN: %clang_cc1 -x hip -triple x86_64-unknown-linux-gnu -aux-triple amdgpu-amd-amdhsa \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM-HOST --implicit-check-not=One --implicit-check-not=Missing
// RUN: %clang_cc1 -x hip -triple amdgpu-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEV --implicit-check-not=Host --implicit-check-not=Missing
// RUN: %clang_cc1 -x hip -triple amdgpu-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM-DEV --implicit-check-not=Host --implicit-check-not=Missing
// RUN: %clang_cc1 -x hip -triple amdgpu-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM-DEV --implicit-check-not=Host --implicit-check-not=Missing

#define __device__ __attribute__((device))
#define __host__ __attribute__((host))

extern "C" {
// Host-only aliasee: emitted on the host only.
// CIR-HOST-DAG: cir.func weak @HostFunc() -> !s32i alias(@__HostFunc)
// CIR-HOST-DAG: cir.func dso_local @HostFunc_() -> !s32i alias(@__HostFunc)
// CIR-HOST-DAG: cir.global weak @HostVar alias(@__HostVar) : !s32i
// CIR-HOST-DAG: cir.global external @HostVar_ alias(@__HostVar) : !s32i
// LLVM-HOST-DAG: @HostFunc = weak alias i32 (), ptr @__HostFunc
// LLVM-HOST-DAG: @HostFunc_ = alias i32 (), ptr @__HostFunc
// LLVM-HOST-DAG: @HostVar = weak alias i32, ptr @__HostVar
// LLVM-HOST-DAG: @HostVar_ = alias i32, ptr @__HostVar
int __HostFunc(void) { return 42; }
int __HostVar = 1;
int HostFunc(void) __attribute__((weak, alias("__HostFunc")));
int HostFunc_(void) __attribute__((alias("__HostFunc")));
extern int __attribute__((weak, alias("__HostVar"))) HostVar;
extern int __attribute__((alias("__HostVar"))) HostVar_;

// Device-only aliasee: emitted on the device only.
// CIR-DEV-DAG: cir.func weak @One() -> !s32i alias(@__One)
// CIR-DEV-DAG: cir.func dso_local @One_() -> !s32i alias(@__One)
// LLVM-DEV-DAG: @One = weak alias i32 (), ptr @__One
// LLVM-DEV-DAG: @One_ = alias i32 (), ptr @__One
__device__ int __One(void) { return 1; }
__device__ int One(void) __attribute__((weak, alias("__One")));
__device__ int One_(void) __attribute__((alias("__One")));

// Host and device aliasee: emitted on both sides.
// CIR-HOST-DAG: cir.func weak @Two() -> !s32i alias(@__Two)
// CIR-HOST-DAG: cir.func dso_local @Two_() -> !s32i alias(@__Two)
// LLVM-HOST-DAG: @Two = weak alias i32 (), ptr @__Two
// LLVM-HOST-DAG: @Two_ = alias i32 (), ptr @__Two
// CIR-DEV-DAG: cir.func weak @Two() -> !s32i alias(@__Two)
// CIR-DEV-DAG: cir.func dso_local @Two_() -> !s32i alias(@__Two)
// LLVM-DEV-DAG: @Two = weak alias i32 (), ptr @__Two
// LLVM-DEV-DAG: @Two_ = alias i32 (), ptr @__Two
__host__ __device__ int __Two(void) { return 2; }
__host__ __device__ int Two(void) __attribute__((weak, alias("__Two")));
__host__ __device__ int Two_(void) __attribute__((alias("__Two")));

// Missing aliasee: skipped on both sides without a diagnostic.
__host__ __device__ int Missing(void) __attribute__((alias("__Missing")));
}
