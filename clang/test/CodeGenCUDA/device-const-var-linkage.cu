// Test that __device__ const variables are externalized only when referenced
// by host code. Variables only used in device code retain internal linkage.

// Non-RDC mode
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -emit-llvm -o - | FileCheck -check-prefix=DEV %s

// RDC mode
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -fgpu-rdc -emit-llvm -o - | FileCheck -check-prefix=RDC %s

// With -fvisibility=hidden
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -fvisibility=hidden -fapply-global-visibility-to-externs \
// RUN:   -emit-llvm -o - | FileCheck -check-prefix=HIDDEN %s

// Negative test: const device vars NOT referenced by host should not be
// externalized.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -emit-llvm -o - | FileCheck -check-prefix=NEG %s

#include "Inputs/cuda.h"

// Case 1: __device__ const referenced by host — should be externalized so
// the host can access it via hipGetSymbolAddress/hipMemcpyFromSymbol. In C++,
// namespace-scope const has internal linkage, but the ODR-use by host code
// triggers externalization to make the symbol visible to the runtime.
// DEV-DAG: @_ZL18const_host_visible = addrspace(4) constant i32 42
// HIDDEN-DAG: @_ZL18const_host_visible = addrspace(4) constant i32 42
// RDC-DAG: @_ZL18const_host_visible.static.{{[0-9a-f_]+}} = addrspace(4) constant i32 42
__device__ const int const_host_visible = 42;

// Case 2: __device__ const NOT referenced by host — should retain internal
// linkage and be optimized away. Only host-referenced const device vars need
// externalization; blindly externalizing all would bloat the symbol table.
// NEG-NOT: @{{.*}}const_dev_only
__device__ const int const_dev_only = 100;

// Case 3: __device__ non-const — always externalized (baseline comparison).
// Non-const device vars already have external linkage by default.
// DEV-DAG: @nonconst_val = addrspace(1) externally_initialized global i32 42
// HIDDEN-DAG: @nonconst_val = protected addrspace(1) externally_initialized global i32 42
__device__ int nonconst_val = 42;

// Case 4: __constant__ const referenced by host — same as Case 1 but in
// constant address space. __constant__ const also gets internal linkage in
// C++ and needs externalization when host code takes its address.
// DEV-DAG: @_ZL17constant_host_ref = addrspace(4) constant i32 200
// HIDDEN-DAG: @_ZL17constant_host_ref = addrspace(4) constant i32 200
// RDC-DAG: @_ZL17constant_host_ref.static.{{[0-9a-f_]+}} = addrspace(4) constant i32 200
__constant__ const int constant_host_ref = 200;

// Case 5: __constant__ const NOT referenced by host — same as Case 2 but
// for __constant__. Should not be externalized.
// NEG-NOT: @{{.*}}constant_no_host
__constant__ const int constant_no_host = 201;

// Case 6: Plain const (no __device__) referenced by host — should NOT be
// externalized on the device side. It gets an implicit CUDAConstantAttr
// (making it CVT_Both) but has no explicit CUDADeviceAttr. The ODR-use
// tracking in SemaExpr checks for an explicit CUDADeviceAttr to distinguish
// this from __device__ const vars.
// NEG-NOT: @{{.*}}plain_const
const int plain_const = 300;

__global__ void kernel(int* out) {
  out[0] = const_host_visible;
  out[1] = const_dev_only;
  out[2] = nonconst_val;
  out[3] = constant_host_ref;
  out[4] = constant_no_host;
  out[5] = plain_const;
}

__host__ __device__ void use(const int *p);
void host_uses() {
  use(&const_host_visible);
  use(&nonconst_val);
  use(&constant_host_ref);
  use(&plain_const);
}

// Verify compiler.used contains the externalized vars.
// DEV: @llvm.compiler.used = {{.*}}@nonconst_val{{.*}}@_ZL1{{[78]}}
// plain_const should NOT be in compiler.used.
// NEG-NOT: @llvm.compiler.used = {{.*}}plain_const
