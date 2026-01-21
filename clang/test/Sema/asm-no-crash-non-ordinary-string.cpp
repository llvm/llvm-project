// RUN: not %clang_cc1 -fsyntax-only -verify %s

void foo() {
  asm("" ::: (u8""}));
  // expected-error@-1 {{cannot use unicode string literal in 'asm'}}
  // expected-error@-2 {{expected ')'}}
}
