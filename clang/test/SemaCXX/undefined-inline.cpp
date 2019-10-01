// RUN: %clang_cc1 -fsyntax-only -triple i686-pc-win32 -verify -std=c++11 %s
// PR14993

namespace test1 {
  inline void f();  // expected-warning{{inline function 'test1::f' is not defined}}
  void test() { f(); }  // expected-note{{used here}}
}

namespace test2 {
  inline int f();
  void test() { (void)sizeof(f()); }
}

namespace test3 {
  void f();  // expected-warning{{inline function 'test3::f' is not defined}}
  inline void f();
  void test() { f(); }  // expected-note{{used here}}
}

namespace test4 {
  inline void error_on_zero(int);    // expected-warning{{inline function 'test4::error_on_zero' is not defined}}
  inline void error_on_zero(char*) {}
  void test() { error_on_zero(0); }  // expected-note{{used here}}
}

namespace test5 {
  struct X { void f(); };
  void test(X &x) { x.f(); }
}

namespace test6 {
  struct X { inline void f(); };  // expected-warning{{inline function 'test6::X::f' is not defined}}
  void test(X &x) { x.f(); }  // expected-note{{used here}}
}

namespace test7 {
  void f();  // expected-warning{{inline function 'test7::f' is not defined}}
  void test() { f(); } // no used-here note.
  inline void f();
}

namespace test8 {
  inline void foo() __attribute__((gnu_inline)); // expected-warning {{'gnu_inline' attribute without 'extern' in C++ treated as externally available, this changed in Clang 10}}
  void test() { foo(); }
}

namespace test9 {
  void foo();
  void test() { foo(); }
  inline void foo() __attribute__((gnu_inline)); // expected-warning {{'gnu_inline' attribute without 'extern' in C++ treated as externally available, this changed in Clang 10}}
}

namespace test10 {
  inline void foo();
  void test() { foo(); }
  inline void foo() __attribute__((gnu_inline)); // expected-warning {{'gnu_inline' attribute without 'extern' in C++ treated as externally available, this changed in Clang 10}}
}

namespace test11 {
  inline void foo() __attribute__((dllexport));
  inline void bar() __attribute__((dllimport));
  void test() { foo(); bar(); }
}

namespace test12 {
  template<typename> constexpr int _S_chk(int *);
  decltype(_S_chk<int>(nullptr)) n;
}
