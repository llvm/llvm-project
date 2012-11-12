// RUN: %clang_cc1 -fsyntax-only -verify %s

int &foo(int); // expected-note {{candidate}}
double &foo(double); // expected-note {{candidate}}
void foo(...) __attribute__((__unavailable__)); // expected-note {{candidate function}} \
// expected-note{{function has been explicitly marked unavailable here}}

void bar(...) __attribute__((__unavailable__)); // expected-note 2{{explicitly marked unavailable}}

void test_foo(short* sp) {
  int &ir = foo(1);
  double &dr = foo(1.0);
  foo(sp); // expected-error{{call to unavailable function 'foo'}}

  void (*fp)(...) = &bar; // expected-error{{'bar' is unavailable}}
  void (*fp2)(...) = bar; // expected-error{{'bar' is unavailable}}

  int &(*fp3)(int) = foo;
  void (*fp4)(...) = foo; // expected-error{{'foo' is unavailable}}
}

namespace radar9046492 {
// rdar://9046492
#define FOO __attribute__((unavailable("not available - replaced")))

void foo() FOO; // expected-note {{candidate function has been explicitly made unavailable}}
void bar() {
  foo(); // expected-error {{call to unavailable function 'foo': not available - replaced}}
}
}

void unavail(short* sp)  __attribute__((__unavailable__));
void unavail(short* sp) {
  // No complains inside an unavailable function.
  int &ir = foo(1);
  double &dr = foo(1.0);
  foo(sp);
  foo();
}
