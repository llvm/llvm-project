// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

template<typename T, typename U> struct is_same {
  static const bool value = false;
};

template<typename T> struct is_same<T, T> {
  static const bool value = true;
};

struct S {
  void f() { static_assert(is_same<decltype(this), S*>::value, ""); }
  void g() const { static_assert(is_same<decltype(this), const S*>::value, ""); }
  void h() volatile { static_assert(is_same<decltype(this), volatile S*>::value, ""); }
  void i() const volatile { static_assert(is_same<decltype(this), const volatile S*>::value, ""); }
};
