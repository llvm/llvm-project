// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// rdar://problem/11120365
namespace test0 {
  template <class T> struct A {
    static void foo(const T &t) {}
    static void foo(T &&t) {
      t.foo(); // expected-error {{member reference base type 'int' is not a structure or union}}
    }
  }; 

  void test() {
    A<int>::foo({}); // expected-note {{requested here}}
  }
}
