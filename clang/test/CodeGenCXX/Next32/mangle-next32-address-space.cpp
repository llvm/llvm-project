// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s

#include <next32_scratchpad.h>

// CHECK-LABEL: define {{.*}}void @_Z3fooPU5NStlsi
void foo(__next32_tls__ int *ptr) {}

// CHECK-LABEL: define {{.*}}void @_Z3barPU8NSglobali
void bar(__next32_global__ int *ptr) {}

// CHECK-LABEL: define {{.*}}void @_Z3zedPU10NSconstanti
void zed(__next32_constant__ int *ptr) {}

// Keyword thread_local is a part of C++11 standard, equivalent to __thread.
// CHECK-LABEL: define {{.*}}ptr addrspace(273) @_ZTW1t
// CHECK-NEXT: %1 = call align 4 ptr addrspace(273) @llvm.threadlocal.address.p273(ptr addrspace(273) align 4 @t)
// CHECK-NEXT: ret ptr addrspace(273) %1
__next32_tls__ thread_local int t = 11;
