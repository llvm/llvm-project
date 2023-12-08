// RUN: %clang_cc1 -std=c++14 %s -emit-llvm -o - -triple nvptx64-nvidia-cuda \
// RUN:   -fcuda-is-device -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++17 %s -emit-llvm -o - -triple nvptx64-nvidia-cuda \
// RUN:   -fcuda-is-device -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++14 %s -emit-llvm -o - \
// RUN:   -triple x86_64-unknown-linux-gnu -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++17 %s -emit-llvm -o - \
// RUN:   -triple x86_64-unknown-linux-gnu -verify -fsyntax-only
#include "Inputs/cuda.h"

template<typename T>
__host__ __device__ void foo(const T **a) {
  // expected-note@-1 {{declared here}}
  static const T b = sizeof(a);
  static constexpr T c = sizeof(a);
  const T d = sizeof(a);
  constexpr T e = sizeof(a);
  constexpr T f = **a;
  // expected-error@-1 {{constexpr variable 'f' must be initialized by a constant expression}}
  // expected-note@-2 {{}}
  a[0] = &b;
  a[1] = &c;
  a[2] = &d;
  a[3] = &e;
}

__device__ void device_fun(const int **a) {
  // expected-note@-1 {{declared here}}
  constexpr int b = sizeof(a);
  static constexpr int c = sizeof(a);
  constexpr int d = **a;
  // expected-error@-1 {{constexpr variable 'd' must be initialized by a constant expression}}
  // expected-note@-2 {{}}
  a[0] = &b;
  a[1] = &c;
  foo(a);
  // expected-note@-1 {{in instantiation of function template specialization 'foo<int>' requested here}}
}

void host_fun(const int **a) {
  // expected-note@-1 {{declared here}}
  constexpr int b = sizeof(a);
  static constexpr int c = sizeof(a);
  constexpr int d = **a;
  // expected-error@-1 {{constexpr variable 'd' must be initialized by a constant expression}}
  // expected-note@-2 {{}}
  a[0] = &b;
  a[1] = &c;
  foo(a);
}

__host__ __device__ void host_device_fun(const int **a) {
  // expected-note@-1 {{declared here}}
  constexpr int b = sizeof(a);
  static constexpr int c = sizeof(a);
  constexpr int d = **a;
  // expected-error@-1 {{constexpr variable 'd' must be initialized by a constant expression}}
  // expected-note@-2 {{}}
  a[0] = &b;
  a[1] = &c;
  foo(a);
}

template <class T>
struct A {
  explicit A() = default;
};
template <class T>
constexpr A<T> a{};

struct B {
  static constexpr bool value = true;
};

template<typename T>
struct C {
  static constexpr bool value = T::value;
};

__constant__ const bool &x = C<B>::value;
