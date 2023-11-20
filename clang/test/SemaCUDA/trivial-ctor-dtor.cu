// RUN: %clang_cc1 -isystem %S/Inputs  -fsyntax-only -verify %s
// RUN: %clang_cc1 -isystem %S/Inputs -fcuda-is-device -fsyntax-only -verify %s

#include <cuda.h>

// Check trivial ctor/dtor
struct A {
  int x;
  A() {}
  ~A() {}
};

__device__ A a;

// Check trivial ctor/dtor of template class
template<typename T>
struct TA {
  T x;
  TA() {}
  ~TA() {}
};

__device__ TA<int> ta;

// Check non-trivial ctor/dtor in parent template class
template<typename T>
struct TB {
  T x;
  TB() { static int nontrivial_ctor = 1; }
  ~TB() {}
};

template<typename T>
struct TC : TB<T> {
  T x;
  TC() {}
  ~TC() {}
};

__device__ TC<int> tc; //expected-error {{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}

// Check trivial ctor specialization
template <typename T>
struct C { //expected-note {{candidate constructor (the implicit copy constructor) not viable}}
           //expected-note@-1 {{candidate constructor (the implicit move constructor) not viable}}
    explicit C() {};
};

template <> C<int>::C() {};
__device__ C<int> ci_d;
C<int> ci_h;

// Check non-trivial ctor specialization
template <> C<float>::C() { static int nontrivial_ctor = 1; } //expected-note {{candidate constructor not viable: call to __host__ function from __device__ function}}
__device__ C<float> cf_d; //expected-error {{no matching constructor for initialization of 'C<float>'}}
C<float> cf_h;
