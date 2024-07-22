// RUN: %clang_cc1 -verify -fsyntax-only %s -Wno-c++17-extensions

int bar();

// expected-note@+1 2 {{conflicting attribute is here}}
[[gnu::always_inline]] void always_inline_fn(void) { }
// expected-note@+1 2 {{conflicting attribute is here}}
[[gnu::flatten]] void flatten_fn(void) { }
[[gnu::noinline]] void noinline_fn(void) { }

void foo() {
  [[clang::noinline]] bar();
  [[clang::noinline(0)]] bar(); // expected-error {{'noinline' attribute takes no arguments}}
  int x;
  [[clang::noinline]] x = 0; // expected-warning {{'noinline' attribute is ignored because there exists no call expression inside the statement}}
  [[clang::noinline]] { asm("nop"); } // expected-warning {{'noinline' attribute is ignored because there exists no call expression inside the statement}}
  [[clang::noinline]] label: x = 1; // expected-warning {{'noinline' attribute only applies to functions and statements}}


  [[clang::noinline]] always_inline_fn(); // expected-warning {{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  [[clang::noinline]] flatten_fn(); // expected-warning {{statement attribute 'noinline' has higher precedence than function attribute 'flatten'}}
  [[clang::noinline]] noinline_fn();

  [[gnu::noinline]] bar(); // expected-warning {{attribute is ignored on this statement as it only applies to functions; use '[[clang::noinline]]' on statements}}
  __attribute__((noinline)) bar(); // expected-warning {{attribute is ignored on this statement as it only applies to functions; use '[[clang::noinline]]' on statements}}
}

void ms_noi_check() {
  [[msvc::noinline]] bar();
  [[msvc::noinline(0)]] bar(); // expected-error {{'noinline' attribute takes no arguments}}
  int x;
  [[msvc::noinline]] x = 0; // expected-warning {{'noinline' attribute is ignored because there exists no call expression inside the statement}}
  [[msvc::noinline]] { asm("nop"); } // expected-warning {{'noinline' attribute is ignored because there exists no call expression inside the statement}}
  [[msvc::noinline]] label: x = 1; // expected-warning {{'noinline' attribute only applies to functions and statements}}

  [[msvc::noinline]] always_inline_fn(); // expected-warning {{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  [[msvc::noinline]] flatten_fn(); // expected-warning {{statement attribute 'noinline' has higher precedence than function attribute 'flatten'}}
  [[msvc::noinline]] noinline_fn();
}

[[clang::noinline]] static int i = bar(); // expected-warning {{'noinline' attribute only applies to functions and statements}}
[[msvc::noinline]] static int j = bar(); // expected-warning {{'noinline' attribute only applies to functions and statements}}

// This used to crash the compiler.
template<int D>
int foo(int x) {
  [[clang::noinline]] return foo<D-1>(x + 1);
}

template<int D>
[[clang::always_inline]]
int dependent(int x){ return x + D;} // #DEP
[[clang::always_inline]]
int non_dependent(int x){return x;} // #NO_DEP

template<int D> [[clang::always_inline]]
int baz(int x) { // #BAZ
  // expected-warning@+2{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  // expected-note@#NO_DEP{{conflicting attribute is here}}
  [[clang::noinline]] non_dependent(x);
  if constexpr (D>0) {
    // expected-warning@+6{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
    // expected-note@#NO_DEP{{conflicting attribute is here}}
    // expected-warning@+4 3{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
    // expected-note@#BAZ 3{{conflicting attribute is here}}
    // expected-note@#BAZ_INST 3{{in instantiation}}
    // expected-note@+1 3{{in instantiation}}
    [[clang::noinline]] return non_dependent(x), baz<D-1>(x + 1);
  }
  return x;
}

// We can't suppress if there is a variadic involved.
template<int ... D>
int variadic_baz(int x) {
  // Diagnoses NO_DEP 2x, once during phase 1, the second during instantiation.
  // Dianoses DEP 3x, once per variadic expansion.
  // expected-warning@+5 2{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  // expected-note@#NO_DEP 2{{conflicting attribute is here}}
  // expected-warning@+3 3{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  // expected-note@#DEP 3{{conflicting attribute is here}}
  // expected-note@#VARIADIC_INST{{in instantiation}}
  [[clang::noinline]] return non_dependent(x) + (dependent<D>(x) + ...);
}

template<int D> [[clang::always_inline]]
int qux(int x) { // #QUX
  // expected-warning@+2{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  // expected-note@#NO_DEP{{conflicting attribute is here}}
  [[msvc::noinline]] non_dependent(x);
  if constexpr (D>0) {
    // expected-warning@+6{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
    // expected-note@#NO_DEP{{conflicting attribute is here}}
    // expected-warning@+4 3{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
    // expected-note@#QUX 3{{conflicting attribute is here}}
    // expected-note@#QUX_INST 3{{in instantiation}}
    // expected-note@+1 3{{in instantiation}}
    [[msvc::noinline]] return non_dependent(x), qux<D-1>(x + 1);
  }
  return x;
}

// We can't suppress if there is a variadic involved.
template<int ... D>
int variadic_qux(int x) {
  // Diagnoses NO_DEP 2x, once during phase 1, the second during instantiation.
  // Dianoses DEP 3x, once per variadic expansion.
  // expected-warning@+5 2{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  // expected-note@#NO_DEP 2{{conflicting attribute is here}}
  // expected-warning@+3 3{{statement attribute 'noinline' has higher precedence than function attribute 'always_inline'}}
  // expected-note@#DEP 3{{conflicting attribute is here}}
  // expected-note@#QUX_VARIADIC_INST{{in instantiation}}
  [[msvc::noinline]] return non_dependent(x) + (dependent<D>(x) + ...);
}

void use() {
  baz<3>(0); // #BAZ_INST
  variadic_baz<0, 1, 2>(0); // #VARIADIC_INST
  qux<3>(0); // #QUX_INST
  variadic_qux<0, 1, 2>(0); // #QUX_VARIADIC_INST
}
