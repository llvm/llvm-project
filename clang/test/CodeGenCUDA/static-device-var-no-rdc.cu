// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefix=DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefix=HOST %s

// Negative tests.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefix=DEV-NEG %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefix=HOST-NEG %s

#include "Inputs/cuda.h"

// Test function scope static device variable, which should not be externalized.
// DEV-DAG: @_ZZ6kernelPiPPKiE1w = internal addrspace(4) constant i32 1
// DEV-DAG: @_ZZ6kernelPiPPKiE21local_static_constant = internal addrspace(4) constant i32 42
// DEV-DAG: @_ZZ6kernelPiPPKiE19local_static_device = internal addrspace(1) constant i32 43

// Check a static device variable referenced by host function is externalized.
// DEV-DAG: @_ZL1x ={{.*}} addrspace(1) externally_initialized global i32 0
// HOST-DAG: @_ZL1x = internal global i32 undef
// HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"_ZL1x\00"

static __device__ int x;

// Check a static device variables referenced only by device functions and kernels
// is not externalized.
// DEV-DAG: @_ZL2x2 = internal addrspace(1) global i32 0
static __device__ int x2;

// Check a static device variable referenced by host device function is externalized.
// DEV-DAG: @_ZL2x3 ={{.*}} addrspace(1) externally_initialized global i32 0
static __device__ int x3;

// Check a static device variable referenced in file scope is externalized.
// DEV-DAG: @_ZL2x4 ={{.*}} addrspace(1) externally_initialized global i32 0
static __device__ int x4;
int& x4_ref = x4;

// Check a static device variable in anonymous namespace.
// DEV-DAG: @_ZN12_GLOBAL__N_12x5E ={{.*}} addrspace(1) externally_initialized global i32 0
namespace {
static __device__ int x5;
}

// Check a static constant variable referenced by host is externalized.
// DEV-DAG: @_ZL1y ={{.*}} addrspace(4) externally_initialized global i32 0
// HOST-DAG: @_ZL1y = internal global i32 undef
// HOST-DAG: @[[DEVNAMEY:[0-9]+]] = {{.*}}c"_ZL1y\00"

static __constant__ int y;

// Test static host variable, which should not be externalized nor registered.
// HOST-DAG: @_ZL1z = internal global i32 0
// DEV-NEG-NOT: @_ZL1z
static int z;

// Test implicit static constant variable, which should not be externalized.
// HOST-DAG: @_ZL2z2 = internal constant i32 456
// DEV-DAG: @_ZL2z2 = internal addrspace(4) constant i32 456

static constexpr int z2 = 456;

// Test static device variable in inline function, which should not be
// externalized nor registered.
// DEV-DAG: @_ZZ6devfunPPKiE1p = linkonce_odr addrspace(4) constant i32 2, comdat

// Check a static device variable referenced by host function only is externalized.
// DEV-DAG: @_ZL1w ={{.*}} addrspace(1) externally_initialized global i32 0
// HOST-DAG: @_ZL1w = internal global i32 undef
// HOST-DAG: @[[DEVNAMEW:[0-9]+]] = {{.*}}c"_ZL1w\00"

static __device__ int w;

// Test non-ODR-use of static device var should not be emitted or registered.
// DEV-NEG-NOT: @_ZL1u
// HOST-NEG-NOT: @_ZL1u

static __device__ int u;

inline __device__ void devfun(const int ** b) {
  const static int p = 2;
  b[0] = &p;
  b[1] = &x2;
}

__global__ void kernel(int *a, const int **b) {
  const static int w = 1;
  const static __constant__ int local_static_constant = 42;
  const static __device__ int local_static_device = 43;
  a[0] = x;
  a[1] = y;
  a[2] = x2;
  a[3] = x3;
  a[4] = x4;
  a[5] = x5;
  a[6] = sizeof(u);
  b[0] = &w;
  b[1] = &z2;
  b[2] = &local_static_constant;
  b[3] = &local_static_device;
  devfun(b);
}

__host__ __device__ void hdf(int *a) {
  a[0] = x3;
}

int* getDeviceSymbol(int *x);

void foo(const int **a) {
  getDeviceSymbol(&x);
  getDeviceSymbol(&x5);
  getDeviceSymbol(&y);
  getDeviceSymbol(&w);
  z = 123;
  a[0] = &z2;
  decltype(u) tmp;
}

// HOST-DAG: __hipRegisterVar({{.*}}@_ZL1x {{.*}}@[[DEVNAMEX]]
// HOST-DAG: __hipRegisterVar({{.*}}@_ZL1y {{.*}}@[[DEVNAMEY]]
// HOST-DAG: __hipRegisterVar({{.*}}@_ZL1w {{.*}}@[[DEVNAMEW]]
// HOST-NEG-NOT: __hipRegisterVar({{.*}}@_ZL1u
// HOST-NEG-NOT: __hipRegisterVar({{.*}}@_ZZ6kernelPiPPKiE1w
// HOST-NEG-NOT: __hipRegisterVar({{.*}}@_ZZ6devfunPPKiE1p
