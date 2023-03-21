// RUN: %clang_cc1 -verify -fsyntax-only %s -Wno-c++17-extensions

int bar();

[[gnu::always_inline]] void always_inline_fn(void) {}
// expected-note@+1{{conflicting attribute is here}}
[[gnu::flatten]] void flatten_fn(void) {}
// expected-note@+1{{conflicting attribute is here}}
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

template<int D>
[[gnu::noinline]]
int dependent(int x){ return x + D;} // #DEP
[[gnu::noinline]]
int non_dependent(int x){return x;} // #NO_DEP

template<int D> [[gnu::noinline]]
int baz(int x) { // #BAZ
  // expected-warning@+2{{statement attribute 'always_inline' has higher precedence than function attribute 'noinline'}}
  // expected-note@#NO_DEP{{conflicting attribute is here}}
  [[clang::always_inline]] non_dependent(x);
  if constexpr (D>0) {
    // expected-warning@+6{{statement attribute 'always_inline' has higher precedence than function attribute 'noinline'}}
    // expected-note@#NO_DEP{{conflicting attribute is here}}
    // expected-warning@+4 3{{statement attribute 'always_inline' has higher precedence than function attribute 'noinline'}}
    // expected-note@#BAZ 3{{conflicting attribute is here}}
    // expected-note@#BAZ_INST 3{{in instantiation}}
    // expected-note@+1 3{{in instantiation}}
    [[clang::always_inline]] return non_dependent(x), baz<D-1>(x + 1);
  }
  return x;
}

// We can't suppress if there is a variadic involved.
template<int ... D>
int variadic_baz(int x) {
  // Diagnoses NO_DEP 2x, once during phase 1, the second during instantiation.
  // Dianoses DEP 3x, once per variadic expansion.
  // expected-warning@+5 2{{statement attribute 'always_inline' has higher precedence than function attribute 'noinline'}}
  // expected-note@#NO_DEP 2{{conflicting attribute is here}}
  // expected-warning@+3 3{{statement attribute 'always_inline' has higher precedence than function attribute 'noinline'}}
  // expected-note@#DEP 3{{conflicting attribute is here}}
  // expected-note@#VARIADIC_INST{{in instantiation}}
  [[clang::always_inline]] return non_dependent(x) + (dependent<D>(x) + ...);
}

void use() {
  baz<3>(0); // #BAZ_INST
  variadic_baz<0, 1, 2>(0); // #VARIADIC_INST

}
