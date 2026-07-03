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

namespace GH26612 {
// This used to crash while issuing the diagnostic about only applying to a
// pointer or reference type.
// FIXME: it would be ideal to only diagnose once rather than twice. We get one
// diagnostic from explicit template arguments and another one for deduced
// template arguments, which seems silly.
template <class T>
void f(T __attribute__((align_value(4))) x) {} // expected-warning 2 {{'align_value' attribute only applies to a pointer or reference ('int' is invalid)}}

void foo() {
  f<int>(0); // expected-note {{while substituting explicitly-specified template arguments into function template 'f'}} \
                expected-note {{while substituting deduced template arguments into function template 'f' [with T = int]}}
}
} // namespace GH26612
