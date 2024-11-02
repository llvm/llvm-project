// RUN: %clang_cc1 -fexperimental-max-bitint-width=1024 -fsyntax-only -verify %s

void f() {
  _Static_assert(__BITINT_MAXWIDTH__ == 1024, "Macro value is unexpected.");

  _BitInt(1024) a;
  unsigned _BitInt(1024) b;

  _BitInt(8388609) c;                // expected-error {{signed _BitInt of bit sizes greater than 1024 not supported}}
  unsigned _BitInt(0xFFFFFFFFFF) d; // expected-error {{unsigned _BitInt of bit sizes greater than 1024 not supported}}
}
