// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

int f() __attribute__((internal_linkage));

class A;
class __attribute__((internal_linkage)) A {
public:
  int x __attribute__((internal_linkage)); // expected-warning{{'internal_linkage' attribute only applies to variables, functions and classes}}
  static int y __attribute__((internal_linkage));
  void f1() __attribute__((internal_linkage));
  void f2() __attribute__((internal_linkage)) {}
  static void f3() __attribute__((internal_linkage)) {}
  void f4(); // expected-note{{previous definition is here}}
  static int zz; // expected-note{{previous definition is here}}
  A() __attribute__((internal_linkage)) {}
  ~A() __attribute__((internal_linkage)) {}
  A& operator=(const A&) __attribute__((internal_linkage)) { return *this; }
  struct {
    int z  __attribute__((internal_linkage)); // expected-warning{{'internal_linkage' attribute only applies to variables, functions and classes}}
  };
};

__attribute__((internal_linkage)) void A::f4() {} // expected-error{{'internal_linkage' attribute does not appear on the first declaration of 'f4'}}

__attribute__((internal_linkage)) int A::zz; // expected-error{{'internal_linkage' attribute does not appear on the first declaration of 'zz'}}

namespace Z __attribute__((internal_linkage)) { // expected-warning{{'internal_linkage' attribute only applies to variables, functions and classes}}
}

__attribute__((internal_linkage("foo"))) int g() {} // expected-error{{'internal_linkage' attribute takes no arguments}}

[[clang::internal_linkage]] int h() {}

enum struct __attribute__((internal_linkage)) E { // expected-warning{{'internal_linkage' attribute only applies to variables, functions and classes}}
  a = 1,
  b = 2
};

int A::y;

void A::f1() {
}

void g(int a [[clang::internal_linkage]]) { // expected-warning{{'internal_linkage' attribute only applies to variables, functions and classes}}
  int x [[clang::internal_linkage]]; // expected-warning{{'internal_linkage' attribute on a non-static local variable is ignored}}
  static int y [[clang::internal_linkage]];
}
