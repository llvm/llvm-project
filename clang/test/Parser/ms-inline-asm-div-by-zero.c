// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fms-extensions -fasm-blocks -fsyntax-only -verify %s

void foo() { // expected-note {{to match this '{'}}
  __asm { return 1 / 0; } // expected-error {{division by zero in assembly expression}}
} // expected-error {{expected '}'}}
