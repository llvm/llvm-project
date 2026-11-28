// RUN: %clang_cc1 -triple spirv64 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -emit-llvm \
// RUN:   -fcuda-is-device -o - %s | FileCheck %s

// Host code may take a kernel's address into an ordinary function pointer,
// as with other targets (for example a table of kernels or a launcher that
// picks one). On HIPSPV the kernel calling convention is applied in CodeGen,
// so the kernels are still emitted as SPIR-V kernels.

#define __global__ __attribute__((global))

typedef void (*kernel_ptr)(float *, float);

// CHECK: define {{.*}}spir_kernel void @_Z3fooPff(
__global__ void foo(float *a, float b) { *a = b; }

// CHECK: define {{.*}}spir_kernel void @_Z3barILi2EEvPff(
template <int N> __global__ void bar(float *a, float b) { *a = b * N; }

// CHECK: define {{.*}}spir_kernel void @_ZL3bazILb1EEvPff(
template <bool B> static __global__ void baz(float *a, float b) {
  *a = B ? b : -b;
}

kernel_ptr pick(int i) {
  kernel_ptr k = foo;
  if (i == 1)
    k = bar<2>;
  else if (i == 2)
    k = &baz<true>;
  return k;
}

template <int N> kernel_ptr pick_t() {
  kernel_ptr k;
  k = bar<N>;
  return k;
}
kernel_ptr p2 = pick_t<2>();
