// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -verify -emit-llvm-only %s

#define __device__ __attribute__((device))
extern "C" __device__ int printf(const char *format, ...);

// Check that we don't crash when asked to printf a non-scalar arg.
struct Struct {
  int x;
  int y;
};

__device__ void PrintfNonScalar(const char *fmt) {
  printf(fmt, 1);
  // Ignore the warning about the %d not matching the struct argument
  // expected-warning@+2 {{}}
  // expected-error@+1 {{cannot compile this non-scalar arg to printf}}
  printf("%d", Struct());
}
