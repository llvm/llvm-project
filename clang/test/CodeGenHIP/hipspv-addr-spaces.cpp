// RUN: %clang_cc1 -triple spirv64 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

// CHECK: %struct.foo_t = type { i32, ptr addrspace(4) }

// CHECK: @d ={{.*}} addrspace(1) externally_initialized global
__device__ int d;

// CHECK: @c ={{.*}} addrspace(1) externally_initialized constant
__constant__ int c;

// CHECK: @s ={{.*}} addrspace(3) global
__shared__ int s;

// CHECK: @foo ={{.*}} addrspace(1) externally_initialized global %struct.foo_t
__device__ struct foo_t {
  int i;
  int* pi;
} foo;

// Check literals are placed in address space 1 (CrossWorkGroup/__global).
// CHECK: @.str ={{.*}} unnamed_addr addrspace(1) constant

// CHECK: define{{.*}} spir_func noundef ptr addrspace(4) @_Z3barPi(ptr addrspace(4)
__device__ int* bar(int *x) {
  return x;
}

// CHECK: define{{.*}} spir_func noundef ptr addrspace(4) @_Z5baz_dv()
__device__ int* baz_d() {
  // CHECK: ret ptr addrspace(4) addrspacecast (ptr addrspace(1) @d to ptr addrspace(4)
  return &d;
}

// CHECK: define{{.*}} spir_func noundef ptr addrspace(4) @_Z5baz_cv()
__device__ int* baz_c() {
  // CHECK: ret ptr addrspace(4) addrspacecast (ptr addrspace(1) @c to ptr addrspace(4)
  return &c;
}

// CHECK: define{{.*}} spir_func noundef ptr addrspace(4) @_Z5baz_sv()
__device__ int* baz_s() {
  // CHECK: ret ptr addrspace(4) addrspacecast (ptr addrspace(3) @s to ptr addrspace(4)
  return &s;
}

// CHECK: define{{.*}} spir_func noundef ptr addrspace(4) @_Z3quzv()
__device__ const char* quz() {
  return "abc";
}
