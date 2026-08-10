// RUN: %clang_cc1 %s -std=c++11 -fsyntax-only -verify

void f1() { return {1,2}; } // expected-error {{void function 'f1' must not return a value}}

template <typename T> void f2() { return {1,2}; } // expected-error {{void function 'f2' must not return a value}}

template <> void f2<float>() { return {1, 2}; } // expected-error {{void function 'f2<float>' must not return a value}}

void test_f2() {
  f2<int>();
  f2<float>();
}

struct S {
  void f3() { return {1,2}; } // expected-error {{void function 'f3' must not return a value}}
  S() { return {1,2}; } // expected-error {{constructor 'S' must not return a value}}
  ~S() { return {1,2}; } // expected-error {{destructor '~S' must not return a value}}
};

template <typename T> struct ST {
  void f4() { return {1,2}; } // expected-error {{void function 'f4' must not return a value}}
  ST() { return {1,2}; } // expected-error {{constructor 'ST<T>' must not return a value}}
  ~ST() { return {1,2}; } // expected-error {{destructor '~ST<T>' must not return a value}}
};

ST<int> st;
