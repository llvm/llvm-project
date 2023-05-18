// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// A glvalue of type "cv1 T1" can be cast to type "rvalue reference to
// cv2 T2" if "cv2 T2" is reference-compatible with "cv1 T1" (8.5.3).
struct A { };
struct B : A { };

template<typename T> T& lvalue();
template<typename T> T&& xvalue();

void test(A &a, B &b) {
  A &&ar0 = static_cast<A&&>(a);
  A &&ar1 = static_cast<A&&>(b);
  A &&ar2 = static_cast<A&&>(lvalue<A>());
  A &&ar3 = static_cast<A&&>(lvalue<B>());
  A &&ar4 = static_cast<A&&>(xvalue<A>());
  A &&ar5 = static_cast<A&&>(xvalue<B>());
  const A &&ar6 = static_cast<const A&&>(a);
  const A &&ar7 = static_cast<const A&&>(b);
  const A &&ar8 = static_cast<const A&&>(lvalue<A>());
  const A &&ar9 = static_cast<const A&&>(lvalue<B>());
  const A &&ar10 = static_cast<const A&&>(xvalue<A>());
  const A &&ar11 = static_cast<const A&&>(xvalue<B>());
}
