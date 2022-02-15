// RUN: %clang_cc1 -triple avr-unknown-unknown -verify %s

const unsigned char val = 0;

int foo(void) {
  __asm__ volatile("foo %0, 1" : : "fo" (val)); // expected-error {{invalid input constraint 'fo' in asm}}
  __asm__ volatile("foo %0, 1" : : "Nd" (val)); // expected-error {{invalid input constraint 'Nd' in asm}}
}
