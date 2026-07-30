// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/std-10-4-ex2-interface.cppm -emit-reduced-module-interface \
// RUN:     -o %t/M.pcm -Wno-unused-value
// RUN: %clang_cc1 -std=c++20 %t/std-10-4-ex2-implementation.cpp -fmodule-file=M=%t/M.pcm \
// RUN:     -fsyntax-only -verify -Wno-unused-value

//--- std-10-4-ex2.h

namespace N {
struct X {};
int d();
int e();
inline int f(X, int = d()) { return e(); }
int g(X);
int h(X);
} // namespace N

//--- std-10-4-ex2-interface.cppm
module;
#include "std-10-4-ex2.h"
export module M;

template <typename T> int use_f() {
  N::X x;           // N::X, N, and  ::  are decl-reachable from use_f
  return f(x, 123); // N::f is decl-reachable from use_f,
                    // N::e is indirectly decl-reachable from use_f
                    //   because it is decl-reachable from N::f, and
                    // N::d is decl-reachable from use_f
                    //   because it is decl-reachable from N::f
                    //   even though it is not used in this call
}

template <typename T> int use_g() {
  N::X x;             // N::X, N, and :: are decl-reachable from use_g
  return g((T(), x)); // N::g is not decl-reachable from use_g
}

template <typename T> int use_h() {
  N::X x;             // N::X, N, and :: are decl-reachable from use_h
  return h((T(), x)); // N::h is not decl-reachable from use_h, but
                      // N::h is decl-reachable from use_h<int>
}

int k = use_h<int>();
// use_h<int> is decl-reachable from k, so
// N::h is decl-reachable from k

//--- std-10-4-ex2-implementation.cpp
module M;

int a = use_f<int>();
int b = use_g<int>();
// expected-error@std-10-4-ex2-interface.cppm:17 {{use of undeclared identifier 'g'}}
// expected-note@-2 {{in instantiation of function template specialization 'use_g<int>' requested here}}
int c = use_h<int>();
