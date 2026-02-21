// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -emit-llvm -o - %s -DSIN_ONLY | FileCheck -check-prefix=NOSINCOS %s

// Test that when a HIP device compilation sees calls to both __ocml_sin and
// __ocml_cos for a given type, Clang injects an __ocml_sincos declaration
// and adds it to @llvm.compiler.used so the demand-linker pulls it in from
// the device library.

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

extern "C" __device__ float __ocml_sin_f32(float);
extern "C" __device__ float __ocml_cos_f32(float);
extern "C" __device__ double __ocml_sin_f64(double);
extern "C" __device__ double __ocml_cos_f64(double);

#ifdef SIN_ONLY
// Only sin, no cos — sincos should NOT be injected.
// NOSINCOS-NOT: __ocml_sincos
__global__ void kernel_sin_only(float *fout, double *dout, float fx, double dx) {
  fout[0] = __ocml_sin_f32(fx);
  dout[0] = __ocml_sin_f64(dx);
}
#else
// Both sin and cos for f32 and f64 — sincos should be injected for both.
// CHECK-DAG: @llvm.compiler.used = {{.*}}@__ocml_sincos_f32{{.*}}@__ocml_sincos_f64
// CHECK-DAG: declare float @__ocml_sincos_f32(float, ptr addrspace(5))
// CHECK-DAG: declare double @__ocml_sincos_f64(double, ptr addrspace(5))
__global__ void kernel_sin_cos_f32_f64(float *fout, double *dout, float fx, double dx) {
  fout[0] = __ocml_sin_f32(fx);
  fout[1] = __ocml_cos_f32(fx);
  dout[0] = __ocml_sin_f64(dx);
  dout[1] = __ocml_cos_f64(dx);
}
#endif
