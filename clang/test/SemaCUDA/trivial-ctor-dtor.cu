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
