// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -D__HIPCC_RTC__ | FileCheck -check-prefixes=CHECK,NOMALLOC %s
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
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -DHIP_VERSION_MAJOR=4 -DHIP_VERSION_MINOR=5 \
// RUN:   -D__HIPCC_RTC__ | FileCheck -check-prefixes=CHECK,MALLOC %s
// RUN: %clang_cc1 -include __clang_hip_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers/cuda_wrappers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -target-cpu gfx906 -emit-llvm %s -fcuda-is-device -o - \
// RUN:   -DHIP_VERSION_MAJOR=4 -DHIP_VERSION_MINOR=5 \
// RUN:   -fsanitize=address -disable-llvm-passes -D__HIPCC_RTC__ \
// RUN:   | FileCheck -check-prefixes=MALLOC-ASAN %s

// expected-no-diagnostics

// Check handling of overriden, implicitly __host__ dtor (should emit as a
// nullptr to global)

struct vbase {
    virtual ~vbase();
};

template<typename T>
struct vderived : public vbase {
    ~vderived();
};

template struct vderived<void>;

// CHECK: @_ZTV8vderivedIvE = weak_odr unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } zeroinitializer, comdat, align 8

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
// CHECK: @_ZTV7derived = linkonce_odr unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @_ZN7derived2pvEv to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @__cxa_deleted_virtual to ptr addrspace(1))] }, comdat, align 8
// CHECK: @_ZTV4base = linkonce_odr unnamed_addr addrspace(1) constant { [4 x ptr addrspace(1)] } { [4 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) null, ptr addrspace(1) addrspacecast (ptr @__cxa_pure_virtual to ptr addrspace(1)), ptr addrspace(1) addrspacecast (ptr @__cxa_deleted_virtual to ptr addrspace(1))] }, comdat, align 8
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
// CXX14: call contract half @llvm.fma.f16
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
// CHECK: call {{.*}}double @llvm.floor.f64(double
__device__ float test_floor() {
  return floor(5);
}

// CHECK-LABEL: define{{.*}}@_Z8test_maxv
// CHECK: call {{.*}}double @llvm.maxnum.f64(double {{.*}}, double
__device__ float test_max() {
  return max(5, 6.0);
}

// CHECK-LABEL: define{{.*}}@_Z10test_isnanv
__device__ double test_isnan() {
  double r = 0;
  double d = 5.0;
  float f = 5.0;

  // AMD_INT_RETURN: call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 3)
  // AMD_BOOL_RETURN: call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 3)
  r += isnan(f);

  // AMD_INT_RETURN: call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 3)
  // AMD_BOOL_RETURN: call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 3)
  r += isnan(d);

  return r ;
}

// Check that device malloc and free do not conflict with std headers.
#include <cstdlib>
// CHECK-LABEL: define{{.*}}@_Z11test_malloc
// CHECK: call {{.*}}ptr @malloc(i64
// CHECK-LABEL: define weak {{.*}}ptr @malloc(i64
// MALLOC:  call i64 @__ockl_dm_alloc
// NOMALLOC:  call void @llvm.trap
// MALLOC-ASAN-LABEL: define weak {{.*}}ptr @malloc(i64
// MALLOC-ASAN:  call ptr @llvm.returnaddress(i32 0)
// MALLOC-ASAN:  call i64 @__asan_malloc_impl(i64 {{.*}}, i64 {{.*}})
__device__ void test_malloc(void *a) {
  a = malloc(42);
}

// CHECK-LABEL: define{{.*}}@_Z9test_free
// CHECK: call {{.*}}void @free(ptr
// CHECK-LABEL: define weak {{.*}}void @free(ptr
// MALLOC:  call void @__ockl_dm_dealloc
// NOMALLOC: call void @llvm.trap
// MALLOC-ASAN-LABEL: define weak {{.*}}void @free(ptr
// MALLOC-ASAN:  call ptr @llvm.returnaddress(i32 0)
// MALLOC-ASAN:  call void @__asan_free_impl(i64 {{.*}}, i64 {{.*}})
__device__ void test_free(void *a) {
  free(a);
}
