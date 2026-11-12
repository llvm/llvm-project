// RUN: %clang_cc1 %s --std=c++11 -triple nvptx-unknown-unknown -fcuda-is-device \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=COMMON,DEVICE
// RUN: %clang_cc1 %s --std=c++11 -triple nvptx-unknown-unknown \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=COMMON,HOST
// RUN: %clang_cc1 %s --std=c++17 -triple nvptx-unknown-unknown -fcuda-is-device \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=COMMON,DEVICE
// RUN: %clang_cc1 %s --std=c++17 -triple nvptx-unknown-unknown \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefixes=COMMON,HOST

#include "Inputs/cuda.h"
extern "C" __host__ void host_fn();
extern "C" __device__ void dev_fn();
extern "C" __host__ __device__ void hd_fn();

// Destructors are handled a bit differently, compared to regular functions.
// Make sure we do trigger kernel generation on the GPU side even if it's only
// referenced by the destructor.
template<typename T> __global__ void f(T) {}
template<typename T> struct A {
  ~A() { f<<<1, 1>>>(T()); }
};

// HOST-LABEL: @a
A<int> a;
// HOST-LABEL: define linkonce_odr void @_ZN1AIiED1Ev
// search further down for the deice-side checks for @_Z1fIiEvT_

struct H1D1 {
  __host__ void operator delete(void *) { host_fn(); };
  __device__ void operator delete(void *) { dev_fn(); };
};

struct H1D2 {
  __host__ void operator delete(void *) { host_fn(); };
  __device__ void operator delete(void *, __SIZE_TYPE__) { dev_fn(); };
};

struct H2D1 {
  __host__ void operator delete(void *, __SIZE_TYPE__) { host_fn(); };
  __device__ void operator delete(void *) { dev_fn(); };
};

struct H2D2 {
  __host__ void operator delete(void *, __SIZE_TYPE__) { host_fn(); };
  __device__ void operator delete(void *, __SIZE_TYPE__) { dev_fn(); };
};

struct H1D1D2 {
  __host__ void operator delete(void *) { host_fn(); };
  __device__ void operator delete(void *) { dev_fn(); };
  __device__ void operator delete(void *, __SIZE_TYPE__) { dev_fn(); };
};

struct H1H2D1 {
  __host__ void operator delete(void *) { host_fn(); };
  __host__ void operator delete(void *, __SIZE_TYPE__) { host_fn(); };
  __device__ void operator delete(void *) { dev_fn(); };
};

struct H1H2D2 {
  __host__ void operator delete(void *) { host_fn(); };
  __host__ void operator delete(void *, __SIZE_TYPE__) { host_fn(); };
  __device__ void operator delete(void *, __SIZE_TYPE__) { dev_fn(); };
};

struct H1H2D1D2 {
  __host__ void operator delete(void *) { host_fn(); };
  __host__ void operator delete(void *, __SIZE_TYPE__) { host_fn(); };
  __device__ void operator delete(void *) { dev_fn(); };
  __device__ void operator delete(void *, __SIZE_TYPE__) { dev_fn(); };
};


template <typename T>
__host__ __device__ void test_hd(void *p) {
  T *t = (T *)p;
  delete t;
}

// Make sure we call the right variant of usual deallocator.
__host__ __device__ void tests_hd(void *t) {
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI4H1D1EvPv
  // COMMON: call void @_ZN4H1D1dlEPv
  test_hd<H1D1>(t);
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI4H1D2EvPv
  // DEVICE: call void @_ZN4H1D2dlEPvj(ptr noundef {{.*}}, i32 noundef 1)
  // HOST:   call void @_ZN4H1D2dlEPv(ptr noundef {{.*}})
  test_hd<H1D2>(t);
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI4H2D1EvPv
  // DEVICE: call void @_ZN4H2D1dlEPv(ptr {{.*}})
  // HOST:   call void @_ZN4H2D1dlEPvj(ptr noundef {{.*}}, i32 noundef 1)
  test_hd<H2D1>(t);
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI4H2D2EvPv
  // COMMON: call void @_ZN4H2D2dlEPvj(ptr noundef {{.*}}, i32 noundef 1)
  test_hd<H2D2>(t);
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI6H1D1D2EvPv
  // COMMON: call void @_ZN6H1D1D2dlEPv(ptr noundef {{.*}})
  test_hd<H1D1D2>(t);
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI6H1H2D1EvPv
  // COMMON: call void @_ZN6H1H2D1dlEPv(ptr {{.*}})
  test_hd<H1H2D1>(t);
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI6H1H2D2EvPv
  // DEVICE: call void @_ZN6H1H2D2dlEPvj(ptr noundef {{.*}}, i32 noundef 1)
  // HOST:   call void @_ZN6H1H2D2dlEPv(ptr noundef {{.*}})
  test_hd<H1H2D2>(t);
  // COMMON-LABEL: define linkonce_odr void @_Z7test_hdI8H1H2D1D2EvPv
  // COMMON: call void @_ZN8H1H2D1D2dlEPv(ptr {{.*}})
  test_hd<H1H2D1D2>(t);
}

// Make sure that we've generated the kernel used by A::~A.
// DEVICE-LABEL: define ptx_kernel void @_Z1fIiEvT_

// Make sure we've picked deallocator for the correct side of compilation.

// COMMON-LABEL: define  linkonce_odr void @_ZN4H1D1dlEPv(ptr noundef %0)
// DEVICE: call void @dev_fn()
// HOST:   call void @host_fn()

// DEVICE-LABEL: define  linkonce_odr void @_ZN4H1D2dlEPvj(ptr noundef %0, i32 noundef %1)
// DEVICE: call void @dev_fn()
// HOST-LABEL: define linkonce_odr void @_ZN4H1D2dlEPv(ptr noundef %0)
// HOST: call void @host_fn()

// DEVICE-LABEL: define  linkonce_odr void @_ZN4H2D1dlEPv(ptr noundef %0)
// DEVICE: call void @dev_fn()
// HOST-LABEL:  define linkonce_odr void @_ZN4H2D1dlEPvj(ptr noundef %0, i32 noundef %1)
// HOST: call void @host_fn()

// COMMON-LABEL: define  linkonce_odr void @_ZN4H2D2dlEPvj(ptr noundef %0, i32 noundef %1)
// DEVICE: call void @dev_fn()
// HOST: call void @host_fn()

// COMMON-LABEL: define  linkonce_odr void @_ZN6H1D1D2dlEPv(ptr noundef %0)
// DEVICE: call void @dev_fn()
// HOST: call void @host_fn()

// COMMON-LABEL: define  linkonce_odr void @_ZN6H1H2D1dlEPv(ptr noundef %0)
// DEVICE: call void @dev_fn()
// HOST: call void @host_fn()

// DEVICE-LABEL: define  linkonce_odr void @_ZN6H1H2D2dlEPvj(ptr noundef %0, i32 noundef %1)
// DEVICE: call void @dev_fn()
// HOST-LABEL: define linkonce_odr void @_ZN6H1H2D2dlEPv(ptr noundef %0)
// HOST: call void @host_fn()

// COMMON-LABEL: define  linkonce_odr void @_ZN8H1H2D1D2dlEPv(ptr noundef %0)
// DEVICE: call void @dev_fn()
// HOST: call void @host_fn()
