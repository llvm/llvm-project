// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ | FileCheck %s
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -include cmath \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ | FileCheck %s  -check-prefixes=AMD_BOOL_RETURN
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -include cmath \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ -DUSE_ISNAN_WITH_INT_RETURN | FileCheck %s -check-prefixes=AMD_INT_RETURN
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ -std=c++14 | FileCheck -check-prefixes=CHECK,CXX14 %s

// expected-no-diagnostics

// Check support for pure and deleted virtual functions
struct base {
  __host__
  __device__
  virtual void pv() = 0;
  __host__
  __device__
  virtual void dv() = delete;
};
struct derived:base {
  __host__
  __device__
  virtual void pv() override {};
};
__device__ void test_vf() {
    derived d;
}
// CHECK: @_ZTV7derived = linkonce_odr unnamed_addr addrspace(1) constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.derived*)* @_ZN7derived2pvEv to i8*), i8* bitcast (void ()* @__cxa_deleted_virtual to i8*)] }, comdat, align 8
// CHECK: @_ZTV4base = linkonce_odr unnamed_addr addrspace(1) constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (void ()* @__cxa_deleted_virtual to i8*)] }, comdat, align 8

// CHECK: define{{.*}}void @__cxa_pure_virtual()
// CHECK: define{{.*}}void @__cxa_deleted_virtual()

struct Number {
  __device__ Number(float _x) : x(_x) {}
  float x;
};

#if __cplusplus >= 201103L
// Check __hip::__numeric_type can be used with a class without default ctor.
__device__ void test_numeric_type() {
  int x = __hip::__numeric_type<Number>::value;
}

// ToDo: Fix __clang_hip_cmake.h to specialize __hip::is_arithmetic<_Float16>
// to resolve fma(_Float16, _Float16, int) to fma(double, double, double)
// instead of fma(_Float16, _Float16, _Float16).

// CXX14-LABEL: define{{.*}}@_Z8test_fma
// CXX14: call {{.*}}@__ocml_fma_f16
__device__ double test_fma(_Float16 h, int i) {
  return fma(h, h, i);
}

#endif

// CHECK-LABEL: amdgpu_kernel void @_Z4kernPff
__global__ void kern(float *x, float y) {
  *x = sin(y);
}

// CHECK-LABEL: define{{.*}} i64 @_Z11test_size_tv
// CHECK: ret i64 8
__device__ size_t test_size_t() {
  return sizeof(size_t);
}

// Check there is no ambiguity when calling overloaded math functions.

// CHECK-LABEL: define{{.*}}@_Z10test_floorv
// CHECK: call {{.*}}double @__ocml_floor_f64(double
__device__ float test_floor() {
  return floor(5);
}

// CHECK-LABEL: define{{.*}}@_Z8test_maxv
// CHECK: call {{.*}}double @__ocml_fmax_f64(double {{.*}}, double
__device__ float test_max() {
  return max(5, 6.0);
}

// CHECK-LABEL: define{{.*}}@_Z10test_isnanv
__device__ double test_isnan() {
  double r = 0;
  double d = 5.0;
  float f = 5.0;

  // AMD_INT_RETURN: call i32 @__ocml_isnan_f32(float
  // AMD_BOOL_RETURN: call i32 @__ocml_isnan_f32(float
  r += isnan(f);

  // AMD_INT_RETURN: call i32 @__ocml_isnan_f64(double
  // AMD_BOOL_RETURN: call i32 @__ocml_isnan_f64(double
  r += isnan(d);

  return r ;
}

// Check that device malloc and free do not conflict with std headers.
#include <cstdlib>
// CHECK-LABEL: define{{.*}}@_Z11test_malloc
// CHECK: call {{.*}}i8* @malloc(i64
// CHECK: define weak {{.*}}i8* @malloc(i64
__device__ void test_malloc(void *a) {
  a = malloc(42);
}

// CHECK-LABEL: define{{.*}}@_Z9test_free
// CHECK: call {{.*}}i8* @free(i8*
// CHECK: define weak {{.*}}i8* @free(i8*
__device__ void test_free(void *a) {
  free(a);
}
