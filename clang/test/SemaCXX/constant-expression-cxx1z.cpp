// RUN: %clang_cc1 -std=c++1z -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

namespace BaseClassAggregateInit {
  struct A {
    int a, b, c;
    constexpr A(int n) : a(n), b(3 * n), c(b - 1) {} // expected-note {{outside the range of representable}}
    constexpr A() : A(10) {};
  };
  struct B : A {};
  struct C { int q; };
  struct D : B, C { int k; };

  constexpr D d1 = { 1, 2, 3 };
  static_assert(d1.a == 1 && d1.b == 3 && d1.c == 2 && d1.q == 2 && d1.k == 3);

  constexpr D d2 = { 14 };
  static_assert(d2.a == 14 && d2.b == 42 && d2.c == 41 && d2.q == 0 && d2.k == 0);

  constexpr D d3 = { A(5), C{2}, 1 };
  static_assert(d3.a == 5 && d3.b == 15 && d3.c == 14 && d3.q == 2 && d3.k == 1);

  constexpr D d4 = {};
  static_assert(d4.a == 10 && d4.b == 30 && d4.c == 29 && d4.q == 0 && d4.k == 0);

  constexpr D d5 = { __INT_MAX__ }; // expected-error {{must be initialized by a constant expression}}
  // expected-note-re@-1 {{in call to 'A({{.*}})'}}
}

namespace NoexceptFunctionTypes {
  template<typename T> constexpr bool f() noexcept(true) { return true; }
  constexpr bool (*fp)() = f<int>;
  static_assert(f<int>());
  static_assert(fp());

  template<typename T> struct A {
    constexpr bool f() noexcept(true) { return true; }
    constexpr bool g() { return f(); }
    constexpr bool operator()() const noexcept(true) { return true; }
  };
  static_assert(A<int>().f());
  static_assert(A<int>().g());
  static_assert(A<int>()());
}

namespace Cxx17CD_NB_GB19 {
  const int &r = 0;
  constexpr int n = r;
}

namespace PR37585 {
template <class T> struct S { static constexpr bool value = true; };
template <class T> constexpr bool f() { return true; }
template <class T> constexpr bool v = true;

void test() {
  if constexpr (true) {}
  else if constexpr (f<int>()) {}
  else if constexpr (S<int>::value) {}
  else if constexpr (v<int>) {}
}
}
