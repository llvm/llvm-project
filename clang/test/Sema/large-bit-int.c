// RUN: %clang_cc1 -triple aarch64-unknown-unknown -fexperimental-max-bitint-width=1024 -fsyntax-only -verify %s

void f() {
  _Static_assert(__BITINT_MAXWIDTH__ == 1024, "Macro value is unexpected.");

  _BitInt(1024) a;
  unsigned _BitInt(1024) b;

  _BitInt(8388609) c;                // expected-error {{signed _BitInt of bit sizes greater than 1024 not supported}}
  unsigned _BitInt(0xFFFFFFFFFF) d; // expected-error {{unsigned _BitInt of bit sizes greater than 1024 not supported}}
}

// Wide elements hit the 2^28 byte limit before the element limit.
typedef _BitInt(1024) too_large_vs __attribute__((vector_size(1 << 29)));     // expected-error {{vector size too large}}
typedef _BitInt(1024) too_large_ev __attribute__((ext_vector_type(1 << 22))); // expected-error {{vector size too large}}
typedef _BitInt(1024) largest_vs __attribute__((vector_size(1 << 28)));
typedef _BitInt(1024) largest_ev __attribute__((ext_vector_type(1 << 21)));
