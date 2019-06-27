// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s


// p3
// A glvalue of type "cv1 T1" can be cast to type "rvalue reference to
// cv2 T2" if "cv2 T2" is reference-compatible with "cv1 T1" (8.5.3).
// p4
// Otherwise, an expression e can be explicitly converted to a type T using a
// static_cast of the form static_cast<T>(e) if the declaration T t(e); is
// well-formed, for some invented temporary variable t (8.5). [...]
struct A { };
struct B : A { };

struct C { explicit operator A&&(); };
struct D { operator B(); };

template<typename T> T& lvalue();
template<typename T> T&& xvalue();
template <typename T> T prvalue();

void test(A &a, B &b) {
  A &&ar0 = static_cast<A&&>(prvalue<A>());
  A &&ar1 = static_cast<A&&>(prvalue<B>());
  A &&ar2 = static_cast<A&&>(lvalue<C>());
  A &&ar3 = static_cast<A&&>(xvalue<C>());
  A &&ar4 = static_cast<A&&>(prvalue<C>());
  A &&ar5 = static_cast<A&&>(lvalue<D>());
  A &&ar6 = static_cast<A&&>(xvalue<D>());
  A &&ar7 = static_cast<A&&>(prvalue<D>());

  A &&ar8 = static_cast<A&&>(prvalue<const A>()); // expected-error {{binding value of type 'const A' to reference to type 'A' drops 'const' qualifier}}
  A &&ar9 = static_cast<A&&>(lvalue<const A>()); // expected-error {{cannot cast from lvalue of type 'const A'}}
  A &&ar10 = static_cast<A&&>(xvalue<const A>()); // expected-error {{cannot cast from rvalue of type 'const A'}}

  const A &&ar11 = static_cast<const A&&>(prvalue<A>());
  const A &&ar12 = static_cast<const A&&>(prvalue<B>());
  const A &&ar13 = static_cast<const A&&>(lvalue<C>());
  const A &&ar14 = static_cast<const A&&>(xvalue<C>());
  const A &&ar15 = static_cast<const A&&>(prvalue<C>());
  const A &&ar16 = static_cast<const A&&>(lvalue<D>());

  const A &&ar17 = static_cast<const A&&>(prvalue<A const volatile>()); // expected-error {{binding value of type 'const volatile A' to reference to type 'const A' drops 'volatile' qualifier}}
}
