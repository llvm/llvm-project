// Check that HIP kernel handles are dso_local according to the rules for global
// variables, with the handle's final linkage.

// Shared library.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -x hip \
// RUN:     -mrelocation-model pic -pic-level 2 \
// RUN:   | FileCheck -check-prefixes=CHECK,DEF-NONLOCAL,DECL-NONLOCAL %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -x hip \
// RUN:     -mrelocation-model pic -pic-level 2 -pic-is-pie \
// RUN:   | FileCheck -check-prefixes=CHECK,DEF-LOCAL,DECL-NONLOCAL %s

// A declared handle may be reached through a copy relocation, which -fno-plt
// does not affect, as the handle is not a function.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -x hip \
// RUN:     -mrelocation-model static \
// RUN:   | FileCheck -check-prefixes=CHECK,DEF-LOCAL,DECL-LOCAL %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -x hip \
// RUN:     -mrelocation-model static -fno-plt \
// RUN:   | FileCheck -check-prefixes=CHECK,DEF-LOCAL,DECL-LOCAL %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - -x hip \
// RUN:     -mrelocation-model static -fno-direct-access-external-data \
// RUN:   | FileCheck -check-prefixes=CHECK,DEF-LOCAL,DECL-NONLOCAL %s

// A declared handle may be auto-imported from a DLL.
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -emit-llvm %s -o - -x hip \
// RUN:   | FileCheck -check-prefixes=CHECK,DEF-LOCAL,DECL-NONLOCAL %s

#include "Inputs/cuda.h"

__global__ void ext_kernel() {}
template <class T> __global__ void tmpl_kernel() {}
template <class T> __global__ void inst_kernel() {}
template __global__ void inst_kernel<float>();
__global__ void decl_kernel();
static __global__ void static_kernel() {}

void launch() {
  ext_kernel<<<1, 1>>>();
  tmpl_kernel<int><<<1, 1>>>();
  decl_kernel<<<1, 1>>>();
  static_kernel<<<1, 1>>>();
}

// DEF-LOCAL-DAG: @_Z10ext_kernelv = dso_local constant ptr @_Z25__device_stub__ext_kernelv, align 8
// DEF-LOCAL-DAG: @_Z11tmpl_kernelIiEvv = linkonce_odr dso_local constant ptr @_Z26__device_stub__tmpl_kernelIiEvv, comdat, align 8
// DEF-LOCAL-DAG: @_Z11inst_kernelIfEvv = weak_odr dso_local constant ptr @_Z26__device_stub__inst_kernelIfEvv, comdat, align 8

// DEF-NONLOCAL-DAG: @_Z10ext_kernelv = constant ptr @_Z25__device_stub__ext_kernelv, align 8
// DEF-NONLOCAL-DAG: @_Z11tmpl_kernelIiEvv = linkonce_odr constant ptr @_Z26__device_stub__tmpl_kernelIiEvv, comdat, align 8
// DEF-NONLOCAL-DAG: @_Z11inst_kernelIfEvv = weak_odr constant ptr @_Z26__device_stub__inst_kernelIfEvv, comdat, align 8

// DECL-LOCAL-DAG: @_Z11decl_kernelv = external dso_local constant ptr, align 8
// DECL-NONLOCAL-DAG: @_Z11decl_kernelv = external constant ptr, align 8

// Internal handles are always dso_local, so it isn't printed.
// CHECK-DAG: @_ZL13static_kernelv = internal constant ptr @_ZL28__device_stub__static_kernelv, align 8
