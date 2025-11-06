// RUN: %clang_cc1 -x hip %s --hipstdpar -triple amdgcn-amd-amdhsa --std=c++17 \
// RUN:   -fcuda-is-device -emit-llvm -o /dev/null -verify

// Note: These would happen implicitly, within the implementation of the
//       accelerator specific algorithm library, and not from user code.

// Calls from the accelerator side to implicitly host (i.e. unannotated)
// functions are fine.

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

extern "C" void host_fn() {}

struct Dummy {};

struct S {
  S() {}
  ~S() { host_fn(); }

  int x;
};

struct T {
  __device__ void hd() { host_fn(); }

  __device__ void hd3();

  void h() {}

  void operator+();
  void operator-(const T&) {}

  operator Dummy() { return Dummy(); }
};

__device__ void T::hd3() { host_fn(); }

template <typename T> __device__ void hd2() { host_fn(); }

__global__ void kernel() { hd2<int>(); }

__device__ void hd() { host_fn(); }

template <typename T> __device__ void hd3() { host_fn(); }
__device__ void device_fn() { hd3<int>(); }

__device__ void local_var() {
  S s;
}

__device__ void explicit_destructor(S *s) {
  s->~S();
}

__device__ void hd_member_fn() {
  T t;

  t.hd();
}

__device__ void h_member_fn() {
  T t;
  t.h();
}

__device__ void unaryOp() {
  T t;
  (void) +t;
}

__device__ void binaryOp() {
  T t;
  (void) (t - t);
}

__device__ void implicitConversion() {
  T t;
  Dummy d = t;
}

template <typename T>
struct TmplStruct {
  template <typename U> __device__ void fn() {}
};

template <>
template <>
__device__ void TmplStruct<int>::fn<int>() { host_fn(); }

__device__ void double_specialization() { TmplStruct<int>().fn<int>(); }
