// RUN: %clang_cc1 -std=c++17 -fsyntax-only -Wno-c++2d-extensions -verify %s
// RUN: %clang_cc1 -std=c++2d -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -Wno-c++2d-extensions -verify -DFP_PRAGMA %s
// RUN: %clang_cc1 -std=c++2d -fsyntax-only -verify -DFP_PRAGMA %s
// expected-no-diagnostics

// A function defaulted outside its class template is not defaulted when its
// declaration is instantiated. Instantiating its definition while non-default
// floating-point features are in effect used to crash.

namespace optnone {
// Instantiating an optnone function leaves its floating-point features in
// effect for the next pending instantiation.
template <typename = int> struct a {
  a();
};
template <typename b> a<b>::a() = default;
template <typename f, typename g> __attribute__((optnone)) void h(f, g) {}
void i(int j) {
  h(j, j);
  a k;
}
} // namespace optnone

namespace special_members {
template <typename T> struct S {
  constexpr S();
  constexpr S(const S &);
  S &operator=(const S &);
};
template <typename T> constexpr S<T>::S() = default;
template <typename T> constexpr S<T>::S(const S &) = default;
template <typename T> S<T> &S<T>::operator=(const S &) = default;

template <typename T> struct D {
  D();
  ~D();
};
template <typename T> D<T>::D() = default;
template <typename T> D<T>::~D() = default;

void f() {
#ifdef FP_PRAGMA
#pragma clang fp contract(fast)
#endif
  // The constexpr constructors are instantiated here, under the pragma.
  constexpr S<int> s;
  constexpr S<int> t = s;
  S<int> u;
  u = t;
  D<int> d;
}
} // namespace special_members

#ifdef FP_PRAGMA
// This stays in effect until the end of the file, where pending instantiations
// are performed.
#pragma clang fp contract(fast)
#endif

namespace postfix {
// The pattern of a postfix operator defaulted outside its class holds the
// result of the prefix operator lookup, which the instantiation reuses.
template <typename T> struct P {
  P &operator++();
  P operator++(int);
};
template <typename T> P<T> P<T>::operator++(int) = default;

// The floating-point features of a postfix operator defaulted outside its
// class are recorded along with the result of the prefix operator lookup.
struct M {
  M &operator++();
  M operator++(int);
};
M M::operator++(int) = default;

struct N {
  N &operator--();
};
N operator--(N &, int) = default;

void g(P<int> p, M m, N n) {
  p++;
  m++;
  n--;
}
} // namespace postfix
