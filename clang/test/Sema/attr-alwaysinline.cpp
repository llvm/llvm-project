// RUN: %clang_cc1 -verify -fsyntax-only %s

int bar();

[[gnu::always_inline]] void always_inline_fn(void) {}
[[gnu::flatten]] void flatten_fn(void) {}

[[gnu::noinline]] void noinline_fn(void) {}

void foo() {
  [[clang::always_inline]] bar();
  [[clang::always_inline(0)]] bar(); // expected-error {{'always_inline' attribute takes no arguments}}
  int x;
  [[clang::always_inline]] int i = bar();  // expected-warning {{'always_inline' attribute only applies to functions and statements}}
  [[clang::always_inline]] x = 0;          // expected-warning {{'always_inline' attribute is ignored because there exists no call expression inside the statement}}
  [[clang::always_inline]] { asm("nop"); } // expected-warning {{'always_inline' attribute is ignored because there exists no call expression inside the statement}}
  [[clang::always_inline]] label : x = 1;  // expected-warning {{'always_inline' attribute only applies to functions and statements}}

  [[clang::always_inline]] always_inline_fn();
  [[clang::always_inline]] noinline_fn(); // expected-warning {{statement attribute 'always_inline' has higher precedence than function attribute 'noinline'}}
  [[clang::always_inline]] flatten_fn();  // expected-warning {{statement attribute 'always_inline' has higher precedence than function attribute 'flatten'}}

  [[gnu::always_inline]] bar();         // expected-warning {{attribute is ignored on this statement as it only applies to functions; use '[[clang::always_inline]]' on statements}}
  __attribute__((always_inline)) bar(); // expected-warning {{attribute is ignored on this statement as it only applies to functions; use '[[clang::always_inline]]' on statements}}
}

[[clang::always_inline]] static int i = bar(); // expected-warning {{'always_inline' attribute only applies to functions and statements}}

// This used to crash the compiler.
template<int D>
int foo(int x) {
  [[clang::always_inline]] return foo<D-1>(x + 1);
}

// FIXME: This should warn that always_inline statement attribute has higher
// precedence than the noinline function attribute.
template<int D> [[gnu::noinline]]
int bar(int x) {
  [[clang::always_inline]] return bar<D-1>(x + 1);
}
