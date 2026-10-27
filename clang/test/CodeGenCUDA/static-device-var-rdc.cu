// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fcuda-is-device \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o %t.nocuid.dev -x hip %s
// RUN: cat %t.nocuid.dev | FileCheck -check-prefixes=DEV,INT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o %t.nocuid.host -x hip %s
// RUN: cat %t.nocuid.host | FileCheck -check-prefixes=HOST,INT-HOST %s

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.dev
// RUN: cat %t.dev | FileCheck -check-prefixes=DEV,EXT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.host
// RUN: cat %t.host | FileCheck -check-prefixes=HOST,EXT-HOST %s

// Check host and device compilations use the same postfixes for static
// variable names.

// RUN: cat %t.dev %t.host | FileCheck -check-prefix=POSTFIX %s
// RUN: cat %t.nocuid.dev %t.nocuid.host | FileCheck -check-prefix=POSTFIX-ID %s

// Negative tests.

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fcuda-is-device \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefix=DEV-NEG %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefix=HOST-NEG %s

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.dev
// RUN: cat %t.dev | FileCheck -check-prefix=DEV-NEG %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.host
// RUN: cat %t.host | FileCheck -check-prefix=HOST-NEG %s

// Check postfix for CUDA.

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - %s | FileCheck \
// RUN:   -check-prefixes=CUDA %s

#include "Inputs/cuda.h"

// Make sure we can still mangle with a line directive.
#line 0 "-"

// Test function scope static device variable, which should not be externalized.
// DEV-DAG: @_ZZ6kernelPiPPKiE1w = internal addrspace(4) constant i32 1


// HOST-DAG: @_ZL1x = internal global i32 undef
// HOST-DAG: @_ZL1y = internal global i32 undef

// Test normal static device variables
// INT-DEV-DAG: @_ZL1x[[FILEID:.*]] = addrspace(1) externally_initialized global i32 0
// INT-HOST-DAG: @.offloading.entry_name{{.*}} = {{.*}}c"_ZL1x[[FILEID:.*]]\00"

// Test externalized static device variables
// EXT-DEV-DAG: @_ZL1x.static.[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// EXT-HOST-DAG: @.offloading.entry_name{{.*}} = {{.*}}c"_ZL1x.static.[[HASH:.*]]\00"
// CUDA-DAG: @_ZL1x__static__[[HASH:.*]] = addrspace(1) externally_initialized global i32 0

// POSTFIX: @_ZL1x.static.[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// POSTFIX: @.offloading.entry_name{{.*}} = {{.*}}c"_ZL1x.static.[[HASH]]\00"
// POSTFIX-ID: @_ZL1x.static.[[FILEID:.*]] = addrspace(1) externally_initialized global i32 0
// POSTFIX-ID: @.offloading.entry_name{{.*}} = {{.*}}c"_ZL1x.static.[[FILEID]]\00"

static __device__ int x;

// Test static device variables not used by host code should not be externalized
// DEV-DAG: @_ZL2x2 = internal addrspace(1) global i32 0

static __device__ int x2;

// Test normal static device variables
// INT-DEV-DAG: @_ZL1y[[FILEID:.*]] = addrspace(4) externally_initialized constant i32 0
// INT-HOST-DAG: @.offloading.entry_name{{.*}} = {{.*}}c"_ZL1y[[FILEID:.*]]\00"

// Test externalized static device variables
// EXT-DEV-DAG: @_ZL1y.static.[[HASH]] = addrspace(4) externally_initialized constant i32 0
// EXT-HOST-DAG: @.offloading.entry_name{{.*}} = {{.*}}c"_ZL1y.static.[[HASH]]\00"

static __constant__ int y;

// Test static host variable, which should not be externalized nor registered.
// HOST-DAG: @_ZL1z = internal global i32 0
// DEV-NEG-NOT: @_ZL1z
static int z;

// Test non-ODR-use of static device variable is not emitted or registered.
// DEV-NEG-NOT: @_ZL1u
// HOST-NEG-NOT: @_ZL1u
static __device__ int u;

// Test static device variable in inline function, which should not be
// externalized nor registered.
// DEV-DAG: @_ZZ6devfunPPKiE1p = linkonce_odr addrspace(4) constant i32 2, comdat

inline __device__ void devfun(const int ** b) {
  const static int p = 2;
  b[0] = &p;
}

__global__ void kernel(int *a, const int **b) {
  const static int w = 1;
  a[0] = x;
  a[1] = y;
  a[2] = sizeof(u);
  b[0] = &w;
  b[1] = &x2;
  devfun(b);
}

int* getDeviceSymbol(int *x);

void foo() {
  getDeviceSymbol(&x);
  getDeviceSymbol(&y);
  z = 123;
  decltype(u) tmp;
}

// The host emits offloading entries for externalized static device variables.
// HOST-DAG: @.offloading.entry._ZL1x.static.{{.*}} = {{.*}}ptr @_ZL1x
// HOST-DAG: @.offloading.entry._ZL1y.static.{{.*}} = {{.*}}ptr @_ZL1y
// HOST-NEG-NOT: @.offloading.entry.{{.*}}_ZL2x2
// HOST-NEG-NOT: @.offloading.entry.{{.*}}_ZZ6kernelPiPPKiE1w
// HOST-NEG-NOT: @.offloading.entry.{{.*}}_ZZ6devfunPPKiE1p
// HOST-NEG-NOT: @.offloading.entry.{{.*}}_ZL1u
