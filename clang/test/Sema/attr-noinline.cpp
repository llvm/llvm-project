// RUN: %clang_cc1 -verify -fsyntax-only %s

int bar();

[[gnu::always_inline]] void always_inline_fn(void) { }
[[gnu::flatten]] void flatten_fn(void) { }

[[gnu::noinline]] void noinline_fn(void) { }

void foo() {
  [[clang::noinline]] bar();
  [[clang::noinline(0)]] bar(); // expected-error {{'noinline' attribute takes no arguments}}
  int x;
  [[clang::noinline]] x = 0; // expected-warning {{'noinline' attribute is ignored because there exists no call expression inside the statement}}
  [[clang::noinline]] { asm("nop"); } // expected-warning {{'noinline' attribute is ignored because there exists no call expression inside the statement}}
  [[clang::noinline]] label: x = 1; // expected-error {{'noinline' attribute only applies to functions and statements}}


  [[clang::noinline]] always_inline_fn(); // expected-warning {{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  [[clang::noinline]] flatten_fn(); // expected-warning {{statement attribute 'noinline' has higher precedence than function attribute 'flatten'}}
  [[clang::noinline]] noinline_fn();

  [[gnu::noinline]] bar(); // expected-warning {{attribute is ignored on this statement as it only applies to functions; use '[[clang::noinline]]' on statements}}
  __attribute__((noinline)) bar(); // expected-warning {{attribute is ignored on this statement as it only applies to functions; use '[[clang::noinline]]' on statements}}
}

[[clang::noinline]] static int i = bar(); // expected-error {{'noinline' attribute only applies to functions and statements}}
