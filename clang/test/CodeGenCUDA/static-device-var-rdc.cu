// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV,INT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST,INT-HOST %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.dev
// RUN: cat %t.dev | FileCheck -check-prefixes=DEV,EXT-DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.host
// RUN: cat %t.host | FileCheck -check-prefixes=HOST,EXT-HOST %s

// Check host and device compilations use the same postfixes for static
// variable names.

// RUN: cat %t.dev %t.host | FileCheck -check-prefix=POSTFIX %s

// Negative tests.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefix=DEV-NEG %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefix=HOST-NEG %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.dev
// RUN: cat %t.dev | FileCheck -check-prefix=DEV-NEG %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -std=c++11 -fgpu-rdc -emit-llvm -o - -x hip %s > %t.host
// RUN: cat %t.host | FileCheck -check-prefix=HOST-NEG %s


#include "Inputs/cuda.h"

// Test function scope static device variable, which should not be externalized.
// DEV-DAG: @_ZZ6kernelPiPPKiE1w = internal addrspace(4) constant i32 1


// HOST-DAG: @_ZL1x = internal global i32 undef
// HOST-DAG: @_ZL1y = internal global i32 undef

// Test normal static device variables
// INT-DEV-DAG: @_ZL1x = addrspace(1) externally_initialized global i32 0
// INT-HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x\00"

// Test externalized static device variables
// EXT-DEV-DAG: @_ZL1x.static.[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// EXT-HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x.static.[[HASH:.*]]\00"

// POSTFIX: @_ZL1x.static.[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// POSTFIX: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x.static.[[HASH]]\00"

static __device__ int x;

// Test static device variables not used by host code should not be externalized
// DEV-DAG: @_ZL2x2 = internal addrspace(1) global i32 0

static __device__ int x2;

// Test normal static device variables
// INT-DEV-DAG: @_ZL1y = addrspace(4) externally_initialized global i32 0
// INT-HOST-DAG: @[[DEVNAMEY:[0-9]+]] = {{.*}}c"_ZL1y\00"

// Test externalized static device variables
// EXT-DEV-DAG: @_ZL1y.static.[[HASH]] = addrspace(4) externally_initialized global i32 0
// EXT-HOST-DAG: @[[DEVNAMEY:[0-9]+]] = {{.*}}c"_ZL1y.static.[[HASH]]\00"

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

// HOST-DAG: __hipRegisterVar({{.*}}@_ZL1x {{.*}}@[[DEVNAMEX]]
// HOST-DAG: __hipRegisterVar({{.*}}@_ZL1y {{.*}}@[[DEVNAMEY]]
// HOST-NEG-NOT: __hipRegisterVar({{.*}}@_ZL2x2
// HOST-NEG-NOT: __hipRegisterVar({{.*}}@_ZZ6kernelPiPPKiE1w
// HOST-NEG-NOT: __hipRegisterVar({{.*}}@_ZZ6devfunPPKiE1p
// HOST-NEG-NOT: __hipRegisterVar({{.*}}@_ZL1u
