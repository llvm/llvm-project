// REQUIRES: amdgpu-registered-target

// What MSVC's <complex> and <cmath> call, from __host__ __device__ functions
// like those of <complex>, with an MSVC host (_MSC_VER and _MSVC_STL_VERSION):
// device code, for HIP and hipRTC, gets the __device__ versions, host code
// still the host functions of the MSVC STL and runtime.

// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.44 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=0 -std=c++17 \
// RUN:   -fcuda-is-device -O1 -emit-llvm %s -o - -verify \
// RUN:   | FileCheck %s --check-prefix=DEV
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.44 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=0 -std=c++17 \
// RUN:   -fcuda-is-device -O1 -emit-llvm %s -o - -verify -D__HIPCC_RTC__ \
// RUN:   | FileCheck %s --check-prefix=DEV
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.44 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=1 -std=c++20 \
// RUN:   -fcuda-is-device -O1 -emit-llvm %s -o - -verify \
// RUN:   | FileCheck %s --check-prefix=DEV
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.44 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=1 -std=c++20 \
// RUN:   -fcuda-is-device -O1 -emit-llvm %s -o - -verify -D__HIPCC_RTC__ \
// RUN:   | FileCheck %s --check-prefix=DEV
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple x86_64-pc-windows-msvc -aux-triple amdgcn-amd-amdhsa \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.44 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=0 -std=c++17 \
// RUN:   -emit-llvm %s -o - -verify | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple x86_64-pc-windows-msvc -aux-triple amdgcn-amd-amdhsa \
// RUN:   -fms-extensions -fms-compatibility -fms-compatibility-version=19.44 \
// RUN:   -D_MSVC_STL_VERSION=144 -D_HAS_CXX20=1 -std=c++20 \
// RUN:   -emit-llvm %s -o - -verify | FileCheck %s --check-prefix=HOST

// expected-no-diagnostics

// Under hipRTC, this <complex> is the first to include <ymath.h>, and must do
// so before it makes the functions of the wrapped <complex> __host__
// __device__.
#include <complex>
#include <utility>

#pragma clang force_cuda_host_device begin

// DEV-LABEL: define{{.*}} i1 @test_signbit(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret i1
// HOST-LABEL: define{{.*}} i1 @test_signbit(
// HOST: call{{.*}} @"?signbit@@YA_NO@Z"(
extern "C" bool test_signbit(long double x) { return std::signbit(x); }

// DEV-LABEL: define{{.*}} i1 @test_isfinite(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret i1
// HOST-LABEL: define{{.*}} i1 @test_isfinite(
// HOST: call{{.*}} @"?isfinite@@YA_NO@Z"(
extern "C" bool test_isfinite(long double x) { return std::isfinite(x); }

// DEV-LABEL: define{{.*}} i1 @test_isinf(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret i1
// HOST-LABEL: define{{.*}} i1 @test_isinf(
// HOST: call{{.*}} @"?isinf@@YA_NO@Z"(
extern "C" bool test_isinf(long double x) { return std::isinf(x); }

// DEV-LABEL: define{{.*}} i1 @test_isnan(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret i1
// HOST-LABEL: define{{.*}} i1 @test_isnan(
// HOST: call{{.*}} @"?isnan@@YA_NO@Z"(
extern "C" bool test_isnan(long double x) { return std::isnan(x); }

// DEV-LABEL: define{{.*}} double @test_fma(
// DEV: call{{.*}} double @llvm.fma.f64(
// HOST-LABEL: define{{.*}} double @test_fma(
// HOST: call{{.*}} @"?fma@@YAOOOO@Z"(
extern "C" long double test_fma(long double x, long double y, long double z) {
  return std::fma(x, y, z);
}

// DEV-LABEL: define{{.*}} i16 @test_exp(
// DEV: call{{.*}} @llvm.exp.f64(
// DEV: call{{.*}} @llvm.ldexp.f64
// HOST-LABEL: define{{.*}} i16 @test_exp(
// HOST: call{{.*}} i16 @_Exp(
extern "C" short test_exp(double *x, double y, short e) {
  return _Exp(x, y, e);
}

// DEV-LABEL: define{{.*}} i16 @test_fexp(
// DEV: call{{.*}} @llvm.exp.f32(
// DEV: call{{.*}} @llvm.ldexp.f32
// HOST-LABEL: define{{.*}} i16 @test_fexp(
// HOST: call{{.*}} i16 @_FExp(
extern "C" short test_fexp(float *x, float y, short e) {
  return _FExp(x, y, e);
}

// DEV-LABEL: define{{.*}} double @test_cosh(
// DEV: call{{.*}} @__ocml_cosh_f64(
// HOST-LABEL: define{{.*}} double @test_cosh(
// HOST: call{{.*}} double @_Cosh(
extern "C" double test_cosh(double x, double y) { return _Cosh(x, y); }

// DEV-LABEL: define{{.*}} float @test_fsinh(
// DEV: call{{.*}} @__ocml_sinh_f32(
// HOST-LABEL: define{{.*}} float @test_fsinh(
// HOST: call{{.*}} float @_FSinh(
extern "C" float test_fsinh(float x, float y) { return _FSinh(x, y); }

// MSVC's std::swap is constexpr only from C++20 on. Before, device code gets
// a __device__ std::swap for types whose moves are trivial.
// DEV-LABEL: define{{.*}} void @test_swap(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret void
// HOST-LABEL: define{{.*}} void @test_swap(
// HOST: call void @"??$swap@N@std@@YAXAEAN0@Z"(
extern "C" void test_swap(double &x, double &y) { std::swap(x, y); }

// DEV-LABEL: define{{.*}} void @test_swap_int(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret void
// HOST-LABEL: define{{.*}} void @test_swap_int(
// HOST: call void @"??$swap@H@std@@YAXAEAH0@Z"(
extern "C" void test_swap_int(int &x, int &y) {
  using std::swap;
  swap(x, y);
}

struct Trivial {
  int i;
  float f;
};
// DEV-LABEL: define{{.*}} void @test_swap_struct(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret void
// HOST-LABEL: define{{.*}} void @test_swap_struct(
// HOST: call void @"??$swap@UTrivial@@@std@@YAXAEAUTrivial@@0@Z"(
extern "C" void test_swap_struct(Trivial &x, Trivial &y) { std::swap(x, y); }

// DEV-LABEL: define{{.*}} void @test_swap_complex(
// DEV-NOT: call{{.*}} @{{[^l]}}
// DEV: ret void
// HOST-LABEL: define{{.*}} void @test_swap_complex(
// HOST: call void @"??$swap@V?$complex@M@std@@@std@@YAXAEAV?$complex@M@0@0@Z"(
extern "C" void test_swap_complex(std::complex<float> &x,
                                  std::complex<float> &y) {
  std::swap(x, y);
}

#pragma clang force_cuda_host_device end

// The moves of NonTrivial are __host__ functions. Device code cannot swap it,
// but a __host__ __device__ function that is not emitted for the device may.
struct NonTrivial {
  NonTrivial(NonTrivial &&);
  NonTrivial &operator=(NonTrivial &&);
};
inline __host__ __device__ void swap_non_trivial(NonTrivial &x, NonTrivial &y) {
  std::swap(x, y);
}

// DEV-NOT: @_Exp(
// DEV-NOT: @_FExp(
// DEV-NOT: @_Cosh(
// DEV-NOT: @_FSinh(
