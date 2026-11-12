// RUN: %clang_cc1 -fsyntax-only -verify %s

void tooManyArgs(void) __attribute__((unavailable("a", "b"))); // expected-error {{'unavailable' attribute takes no more than 1 argument}}

int foo(int)  __attribute__((__unavailable__("USE IFOO INSTEAD"))); // expected-note {{'foo' has been explicitly marked unavailable here}}
double dfoo(double)  __attribute__((__unavailable__("NO LONGER"))); // expected-note 2 {{'dfoo' has been explicitly marked unavailable here}}

void bar(void) __attribute__((__unavailable__)); // expected-note {{explicitly marked unavailable}}

int quux(void) __attribute__((__unavailable__(12))); // expected-error {{expected string literal as argument of '__unavailable__' attribute}}

#define ACCEPTABLE	"Use something else"
int quux2(void) __attribute__((__unavailable__(ACCEPTABLE)));

void test_foo(void) {
  int ir = foo(1); // expected-error {{'foo' is unavailable: USE IFOO INSTEAD}}
  double dr = dfoo(1.0); // expected-error {{'dfoo' is unavailable: NO LONGER}}

  void (*fp)(void) = &bar; // expected-error {{'bar' is unavailable}}

  double (*fp4)(double) = dfoo;  // expected-error {{'dfoo' is unavailable: NO LONGER}}
}

char test2[__has_feature(attribute_unavailable_with_message) ? 1 : -1];

void unavail(void)  __attribute__((__unavailable__));
void unavail(void) {
  // No complains inside an unavailable function.
  int ir = foo(1);
  double dr = dfoo(1.0);
  void (*fp)(void) = &bar;
  double (*fp4)(double) = dfoo;
}

enum foo {
    a = 1,
    b __attribute__((deprecated())) = 2, // expected-note {{'b' has been explicitly marked deprecated here}}
    c = 3
}__attribute__((deprecated())); // expected-note {{'foo' has been explicitly marked deprecated here}}

enum fee { // expected-note 2 {{'fee' has been explicitly marked unavailable here}}
    r = 1,
    s = 2,
    t = 3
}__attribute__((unavailable()));

enum fee f(void) { // expected-error {{'fee' is unavailable}}
    int i = a; // expected-warning {{'a' is deprecated}}

    i = b; // expected-warning {{'b' is deprecated}}

    return r; // expected-error {{'r' is unavailable}}
}
