// Test that __device__ const variables are emitted and preserved so they can be
// found by hipGetSymbolAddress. For CUDA compatibility, __device__ const
// variables should be externalized and added to llvm.compiler.used.

// Non-RDC mode
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -emit-llvm -o - | FileCheck -check-prefix=DEV %s

// RDC mode - symbols should still be externalized and preserved
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -fgpu-rdc -emit-llvm -o - | FileCheck -check-prefix=RDC %s

// With -fvisibility=hidden (default for HIP device compilation), externalized
// __device__ const variables remain visible to HSA runtime symbol lookup.
// Non-const device variables get protected visibility; const device variables
// get default visibility (which is also sufficient for symbol lookup).
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -fvisibility=hidden -fapply-global-visibility-to-externs \
// RUN:   -emit-llvm -o - | FileCheck -check-prefix=HIDDEN %s

#include "Inputs/cuda.h"

// A __device__ const variable used in a kernel.
// It should be emitted with external linkage (no 'internal' keyword) and
// added to @llvm.compiler.used so hipGetSymbolAddress can find it.
// The name is mangled with _ZL prefix due to C++ const rules, but it has
// external linkage in LLVM IR (no 'internal' keyword).

// DEV: @_ZL9const_val = addrspace(4) externally_initialized constant i32 42
// HIDDEN: @_ZL9const_val = addrspace(4) externally_initialized constant i32 42
__device__ const int const_val = 42;

// A __device__ non-const variable for comparison - this works correctly
// DEV: @nonconst_val = addrspace(1) externally_initialized global i32 42
// HIDDEN: @nonconst_val = protected addrspace(1) externally_initialized global i32 42
__device__ int nonconst_val = 42;

// A static __device__ const should also be externalized
// DEV: @_ZL16static_const_val = addrspace(4) externally_initialized constant i32 100
// HIDDEN: @_ZL16static_const_val = addrspace(4) externally_initialized constant i32 100
static __device__ const int static_const_val = 100;

// Kernel that uses the variables to ensure they're not optimized away
__global__ void kernel(int* out) {
  out[0] = const_val;
  out[1] = nonconst_val;
  out[2] = static_const_val;
}

// All device variables should be in llvm.compiler.used on device side
// DEV: @llvm.compiler.used = {{.*}}@_ZL9const_val{{.*}}@nonconst_val{{.*}}@_ZL16static_const_val

// RDC mode checks - symbols have unique suffixes but should still be externalized
// RDC: @_ZL9const_val.static.{{[0-9a-f_]+}} = addrspace(4) externally_initialized constant i32 42
// RDC: @nonconst_val = addrspace(1) externally_initialized global i32 42
// RDC: @_ZL16static_const_val.static.{{[0-9a-f_]+}} = addrspace(4) externally_initialized constant i32 100
// RDC: @llvm.compiler.used = {{.*}}@_ZL9const_val{{.*}}@nonconst_val{{.*}}@_ZL16static_const_val
