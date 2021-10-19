// RUN: %clang_cc1 -verify -fsyntax-only %s

void bar();

void foo() {
  [[clang::nomerge]] bar();
  [[clang::nomerge(1, 2)]] bar(); // expected-error {{'nomerge' attribute takes no arguments}}
  int x;
  [[clang::nomerge]] x = 10; // expected-warning {{nomerge attribute is ignored because there exists no call expression inside the statement}}

  [[clang::nomerge]] label: bar(); // expected-error {{'nomerge' attribute only applies to functions and statements}}

}

[[clang::nomerge]] int f();

[[clang::nomerge]] static int i = f(); // expected-error {{'nomerge' attribute only applies to functions and statements}}
