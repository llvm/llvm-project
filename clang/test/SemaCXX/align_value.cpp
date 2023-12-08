// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef double * __attribute__((align_value(64))) aligned_double;

void foo(aligned_double x, double * y __attribute__((align_value(32))),
         double & z __attribute__((align_value(128)))) { };

template <typename T, int Q>
struct x {
  typedef T* aligned_int __attribute__((align_value(32+8*Q)));
  aligned_int V;

  void foo(aligned_int a, T &b __attribute__((align_value(sizeof(T)*4))));
};

x<float, 4> y;

template <typename T, int Q>
struct nope {
  // expected-error@+1 {{requested alignment is not a power of 2}}
  void foo(T &b __attribute__((align_value(sizeof(T)+1))));
};

// expected-note@+1 {{in instantiation of template class 'nope<long double, 4>' requested here}}
nope<long double, 4> y2;

