// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-linux-gnu %s

int x __attribute__((section(
   42)));  // expected-error {{'section' attribute requires a string}}


// PR6007
void test() {
  __attribute__((section("NEAR,x"))) int n1; // expected-error {{'section' attribute only applies to functions, global variables, Objective-C methods, and Objective-C properties}}
  __attribute__((section("NEAR,x"))) static int n2; // ok.
}

// pr9356
void __attribute__((section("foo"))) test2(); // expected-note {{previous attribute is here}}
void __attribute__((section("bar"))) test2() {} // expected-warning {{section does not match previous declaration}}

enum __attribute__((section("NEAR,x"))) e { one }; // expected-error {{'section' attribute only applies to}}

extern int a; // expected-note {{previous declaration is here}}
int *b = &a;
extern int a __attribute__((section("foo,zed"))); // expected-warning {{section attribute is specified on redeclared variable}}

// Not a warning.
extern int c;
int c __attribute__((section("foo,zed")));

// Also OK.
struct r_debug {};
extern struct r_debug _r_debug;
struct r_debug _r_debug __attribute__((nocommon, section(".r_debug,bar")));

namespace override {
  struct A {
    __attribute__((section("foo"))) virtual void f(){};
  };
  struct B : A {
    void f() {} // ok
  };
  struct C : A {
    __attribute__((section("bar"))) void f(); // expected-note {{previous}}
  };
  __attribute__((section("baz"))) // expected-warning {{section does not match}}
  void C::f() {}
}
