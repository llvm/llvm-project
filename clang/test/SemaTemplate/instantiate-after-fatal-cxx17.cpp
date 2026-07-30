// RUN: not %clang_cc1 -std=c++17 -fsyntax-only -ferror-limit 1 %s 2>&1 | FileCheck %s

#error Error 1
#error Error 2
// CHECK: fatal error: too many errors emitted, stopping now

namespace rdar39051732 {

  template<class T> struct A {
    template <class U> A(T&, ...);
  };
  // Deduction guide triggers constructor instantiation.
  template<class T> A(const T&, const T&) -> A<T&>;

}

