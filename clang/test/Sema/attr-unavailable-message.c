// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: //6734520

int foo(int)  __attribute__((__unavailable__("USE IFOO INSTEAD"))); // expected-note {{function has been explicitly marked unavailable here}}
double dfoo(double)  __attribute__((__unavailable__("NO LONGER"))); // expected-note 2 {{function has been explicitly marked unavailable here}}

void bar() __attribute__((__unavailable__)); // expected-note {{explicitly marked unavailable}}

void test_foo() {
  int ir = foo(1); // expected-error {{'foo' is unavailable: USE IFOO INSTEAD}}
  double dr = dfoo(1.0); // expected-error {{'dfoo' is unavailable: NO LONGER}}

  void (*fp)() = &bar; // expected-error {{'bar' is unavailable}}

  double (*fp4)(double) = dfoo;  // expected-error {{'dfoo' is unavailable: NO LONGER}}
}

char test2[__has_feature(attribute_unavailable_with_message) ? 1 : -1];

// rdar://9623855
void unavail(void)  __attribute__((__unavailable__));
void unavail(void) {
  // No complains inside an unavailable function.
  int ir = foo(1);
  double dr = dfoo(1.0);
  void (*fp)() = &bar;
  double (*fp4)(double) = dfoo;
}

// rdar://10201690
enum foo {
    a = 1, // expected-note {{declared here}}
    b __attribute__((deprecated())) = 2, // expected-note {{declared here}}
    c = 3
}__attribute__((deprecated()));  

enum fee { // expected-note {{declaration has been explicitly marked unavailable here}}
    r = 1, // expected-note {{declaration has been explicitly marked unavailable here}}
    s = 2,
    t = 3
}__attribute__((unavailable()));  

enum fee f() { // expected-error {{'fee' is unavailable}}
    int i = a; // expected-warning {{'a' is deprecated}}

    i = b; // expected-warning {{'b' is deprecated}}

    return r; // expected-error {{'r' is unavailable}}
}
