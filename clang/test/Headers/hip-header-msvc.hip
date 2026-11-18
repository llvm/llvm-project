// REQUIRES: amdgpu-registered-target

// Device versions of what MSVC's <complex> calls, with an MSVC host (_MSC_VER
// and _MSVC_STL_VERSION), for HIP and hipRTC.

// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.33 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=0 -std=c++17 \
// RUN:   -fcuda-is-device -O1 -emit-llvm %s -o - -verify | FileCheck %s
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.33 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=0 -std=c++17 \
// RUN:   -fcuda-is-device -O1 -emit-llvm %s -o - -verify \
// RUN:   -D__HIPCC_RTC__ | FileCheck %s

// expected-no-diagnostics

// Under hipRTC, this <complex> is the first to include <ymath.h>, and must do
// so before it makes the functions of the wrapped <complex> __host__
// __device__.
#include <complex>

// CHECK-LABEL: define{{.*}} i1 @test_signbit_ld(
// CHECK-NOT: call
// CHECK: ret i1
extern "C" __device__ bool test_signbit_ld(long double x) {
  return signbit(x);
}

// CHECK-LABEL: define{{.*}} double @test_fma_ld(
// CHECK: call{{.*}} double @llvm.fma.f64(
extern "C" __device__ long double test_fma_ld(long double x, long double y,
                                              long double z) {
  return fma(x, y, z);
}

// CHECK-LABEL: define{{.*}} i16 @test_exp(
// CHECK: call{{.*}} @llvm.exp.f64(
// CHECK: call{{.*}} @llvm.ldexp.f64
extern "C" __device__ short test_exp(double *x, double y, short e) {
  return _Exp(x, y, e);
}

// CHECK-LABEL: define{{.*}} i16 @test_fexp(
// CHECK: call{{.*}} @llvm.exp.f32(
// CHECK: call{{.*}} @llvm.ldexp.f32
extern "C" __device__ short test_fexp(float *x, float y, short e) {
  return _FExp(x, y, e);
}

// CHECK-LABEL: define{{.*}} double @test_cosh(
// CHECK: call{{.*}} @__ocml_cosh_f64(
extern "C" __device__ double test_cosh(double x, double y) {
  return _Cosh(x, y);
}

// CHECK-LABEL: define{{.*}} float @test_fsinh(
// CHECK: call{{.*}} @__ocml_sinh_f32(
extern "C" __device__ float test_fsinh(float x, float y) {
  return _FSinh(x, y);
}

// CHECK-LABEL: define{{.*}} void @test_swap(
// CHECK-NOT: call
// CHECK: ret void
extern "C" __device__ void test_swap(double &x, double &y) {
  std::swap(x, y);
}

// CHECK-NOT: @_Exp(
// CHECK-NOT: @_FExp(
// CHECK-NOT: @_Cosh(
// CHECK-NOT: @_FSinh(
