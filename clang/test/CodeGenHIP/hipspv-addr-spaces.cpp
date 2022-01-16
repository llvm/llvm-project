// RUN: %clang_cc1 -triple spirv64 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

// CHECK: %struct.foo_t = type { i32, i32 addrspace(4)* }

// CHECK: @d ={{.*}} addrspace(1) externally_initialized global
__device__ int d;

// CHECK: @c ={{.*}} addrspace(1) externally_initialized global
__constant__ int c;

// CHECK: @s ={{.*}} addrspace(3) global
__shared__ int s;

// CHECK: @foo ={{.*}} addrspace(1) externally_initialized global %struct.foo_t
__device__ struct foo_t {
  int i;
  int* pi;
} foo;

// CHECK: define{{.*}} spir_func noundef i32 addrspace(4)* @_Z3barPi(i32 addrspace(4)*
__device__ int* bar(int *x) {
  return x;
}

// CHECK: define{{.*}} spir_func noundef i32 addrspace(4)* @_Z5baz_dv()
__device__ int* baz_d() {
  // CHECK: ret i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @d to i32 addrspace(4)*
  return &d;
}

// CHECK: define{{.*}} spir_func noundef i32 addrspace(4)* @_Z5baz_cv()
__device__ int* baz_c() {
  // CHECK: ret i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @c to i32 addrspace(4)*
  return &c;
}

// CHECK: define{{.*}} spir_func noundef i32 addrspace(4)* @_Z5baz_sv()
__device__ int* baz_s() {
  // CHECK: ret i32 addrspace(4)* addrspacecast (i32 addrspace(3)* @s to i32 addrspace(4)*
  return &s;
}
