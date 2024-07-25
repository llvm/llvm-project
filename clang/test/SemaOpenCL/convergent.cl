// RUN: %clang_cc1 -triple spir-unknown-unknown -fsyntax-only -verify %s

void f1(void) __attribute__((convergent));

void f2(void) __attribute__((convergent(1))); // expected-error {{'convergent' attribute takes no arguments}}

void f3(int a __attribute__((convergent))); // expected-warning {{'convergent' attribute only applies to functions and statements}}

void f4(void) {
  int var1 __attribute__((convergent)); // expected-warning {{'convergent' attribute only applies to functions and statements}}
}

void bar();

void foo() {
  [[clang::convergent]] bar();
  [[clang::convergent(1, 2)]] bar(); // expected-error {{'convergent' attribute takes no arguments}}
  int x;
  [[clang::convergent]] x = 10; // expected-warning {{'convergent' attribute is ignored because there exists no call expression inside the statement}}

  [[clang::convergent]] label: bar(); // expected-warning {{'convergent' attribute only applies to functions and statements}}

}
