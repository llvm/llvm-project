// RUN: %clang_cc1 -triple aarch64-unknown-none-gnu %s -verify -fsyntax-only

register char i1 __asm__ ("x15"); // expected-error {{size of register 'x15' does not match variable size}}
register long long l2 __asm__ ("w14"); // expected-error {{size of register 'w14' does not match variable size}}
