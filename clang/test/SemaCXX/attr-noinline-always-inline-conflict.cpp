// RUN: %clang_cc1 %s -std=c++17 -fsyntax-only -verify

void foo() {}

// Statement attributes are mutually exclusive
void caller() {
  [[clang::noinline, clang::always_inline]] foo(); // expected-error {{'clang::always_inline' and 'clang::noinline' attributes are not compatible}} \
                                                   // expected-note {{conflicting attribute is here}}

  [[clang::always_inline, clang::noinline]] foo(); // expected-error {{'clang::noinline' and 'clang::always_inline' attributes are not compatible}} \
                                                   // expected-note {{conflicting attribute is here}}
}

// Attributes on redeclared functions are mutually exclusive
[[clang::noinline]] void redecl_fn(); // expected-note {{conflicting attribute is here}}
[[clang::always_inline]] void redecl_fn() {} // expected-error {{'clang::always_inline' and 'clang::noinline' attributes are not compatible}}

[[clang::always_inline]] void redecl_fn2(); // expected-note {{conflicting attribute is here}}
[[clang::noinline]] void redecl_fn2() {} // expected-error {{'clang::noinline' and 'clang::always_inline' attributes are not compatible}}

// Attributes on the same declaration are mutually exclusive
[[clang::noinline, clang::always_inline]] void decl_fn(); // expected-error {{'clang::always_inline' and 'clang::noinline' attributes are not compatible}} \
                                                          // expected-note {{conflicting attribute is here}}

// Explicit specialization should not inherit inline attributes
template <typename T>
[[clang::noinline]] void tmpl_fn(T);

template <>
[[clang::always_inline]] void tmpl_fn(int); // no error expected

// Check different spellings
[[gnu::noinline]] void spelling_fn(); // expected-note {{conflicting attribute is here}}
[[gnu::always_inline]] void spelling_fn() {} // expected-error {{'gnu::always_inline' and 'gnu::noinline' attributes are not compatible}}

__attribute__((noinline)) void spelling_fn2(); // expected-note {{conflicting attribute is here}}
__attribute__((always_inline)) void spelling_fn2() {} // expected-error {{'always_inline' and 'noinline' attributes are not compatible}}
