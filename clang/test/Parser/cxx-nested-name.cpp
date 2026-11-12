// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace binary_operator {
  namespace N {
    template <class> struct A {
      static const int y = 0;
    };
  } // namespace N
  void f(int x) { (void)(x < N::A<int>::y); }
} // namespace binary_operator
