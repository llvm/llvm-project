// Test that template argument deduction is deferred correctly.
//
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -verify-ignore-unexpected=note %s

#include "Inputs/cuda.h"

namespace h_free_call {
  template<class T>
  concept DoNotDeduct = []() {
    static_assert(sizeof(T) == 0);
    return true;
  }();

  void fn(int) {}
  void fn(DoNotDeduct auto) {}

  void call() {
    fn(0);
    fn(nullptr); // expected-error@-9 {{static assertion failed due to requirement 'sizeof(std::nullptr_t) == 0'}}
  }
}

namespace h_member_call {
  template<class T>
  concept DoNotDeduct = []() {
    static_assert(sizeof(T) == 0);
    return true;
  }();

  struct A {
    void operator=(int) {}
    void operator=(DoNotDeduct auto) {}
  };

  void call(A a) {
    a.operator=(0);
    a.operator=(nullptr); // expected-error@-11 {{static assertion failed due to requirement 'sizeof(std::nullptr_t) == 0'}}
  }
}

namespace hd_free_call {
  template<class T>
  concept DoNotDeduct = []() {
    static_assert(sizeof(T) == 0);
    return true;
  }();

  __host__ __device__ void fn(int) {}
  __host__ __device__ void fn(DoNotDeduct auto) {}

  __host__ __device__ void call() {
    fn(0); // expected-error@-8 {{static assertion failed due to requirement 'sizeof(int) == 0'}}
    fn(nullptr); // expected-error@-9 {{static assertion failed due to requirement 'sizeof(std::nullptr_t) == 0'}}
  }
}

namespace hd_member_call {
  template<class T>
  concept DoNotDeduct = []() {
    static_assert(sizeof(T) == 0);
    return true;
  }();

  struct A {
    __host__ __device__ void operator=(int) {}
    __host__ __device__ void operator=(DoNotDeduct auto) {}
  };

  __host__ __device__ void call(A a) {
    a.operator=(0); // expected-error@-10 {{static assertion failed due to requirement 'sizeof(int) == 0'}}
    a.operator=(nullptr); // expected-error@-11 {{static assertion failed due to requirement 'sizeof(std::nullptr_t) == 0'}}
  }
}
