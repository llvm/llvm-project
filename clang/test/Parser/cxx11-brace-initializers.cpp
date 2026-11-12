// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

struct S {
  S(int, int) {}
};

void f(int, S const&, int) {}

void test1()
{
  S X1{1, 1,};
  S X2 = {1, 1,};

  f(0, {1, 1}, 0);
}

namespace PR14948 {
  template<typename T> struct Q { static T x; };

  struct X {};
  template<> X Q<X>::x {};
  template<> int Q<int[]>::x[] { 1, 2, 3 };
  template<> int Q<int>::x { 1 };

  template<typename T> T Q<T>::x {};
}

int conditional1 = 1 ? {} : 0; // expected-error {{initializer list cannot be used on the right hand side of operator '?'}}
int conditional2 = 1 ? 0 : {}; // expected-error {{initializer list cannot be used on the right hand side of operator ':'}}
