// RUN: %clang_cc1 %s --std=c++11 -triple nvptx-unknown-unknown -fcuda-is-device \
// RUN:   -emit-llvm -o /dev/null -verify=device
// RUN: %clang_cc1 %s --std=c++11 -triple nvptx-unknown-unknown \
// RUN:   -emit-llvm -o /dev/null -verify=host
// RUN: %clang_cc1 %s --std=c++17 -triple nvptx-unknown-unknown -fcuda-is-device \
// RUN:   -emit-llvm -o /dev/null -verify=device
// RUN: %clang_cc1 %s --std=c++17 -triple nvptx-unknown-unknown \
// RUN:   -emit-llvm -o /dev/null -verify=host

#include "Inputs/cuda.h"
extern __host__ void host_fn();
extern __device__ void dev_fn();
extern __host__ __device__ void hd_fn();

struct H1D1 {
  __host__ void operator delete(void *) { host_fn(); };
  __device__ void operator delete(void *) { dev_fn(); };
};

struct h1D1 {
  __host__ void operator delete(void *) = delete;
  // host-note@-1 {{'operator delete' has been explicitly marked deleted here}}
  __device__ void operator delete(void *) { dev_fn(); };
};

struct H1d1 {
  __host__ void operator delete(void *) { host_fn(); };
  __device__ void operator delete(void *) = delete;
  // device-note@-1 {{'operator delete' has been explicitly marked deleted here}}
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
  // host-error@-1 {{attempt to use a deleted function}}
  // device-error@-2 {{attempt to use a deleted function}}
}

__host__ __device__ void tests_hd(void *t) {
  test_hd<H1D1>(t);
  test_hd<h1D1>(t);
  // host-note@-1 {{in instantiation of function template specialization 'test_hd<h1D1>' requested here}}
  test_hd<H1d1>(t);
  // device-note@-1 {{in instantiation of function template specialization 'test_hd<H1d1>' requested here}}
  test_hd<H1D2>(t);
  test_hd<H2D1>(t);
  test_hd<H2D2>(t);
  test_hd<H1D1D2>(t);
  test_hd<H1H2D1>(t);
  test_hd<H1H2D1>(t);
  test_hd<H1H2D2>(t);
  test_hd<H1H2D1D2>(t);
}

// This should produce no errors.  Defaulted destructor should be treated as HD,
// which allows referencing host-only `operator delete` with a deferred
// diagnostics that would fire if we ever attempt to codegen it on device..
struct H {
  virtual ~H() = default;
  static void operator delete(void *) {}
};
H h;
