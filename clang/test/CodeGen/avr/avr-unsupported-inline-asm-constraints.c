// RUN: %clang_cc1 -triple avr-unknown-unknown -verify %s

const unsigned char val = 0;

int foo(void) {
  __asm__ volatile("foo %0, 1" : : "fo" (val)); // expected-error {{invalid input constraint 'fo' in asm}}
  __asm__ volatile("foo %0, 1" : : "Nd" (val)); // expected-error {{invalid input constraint 'Nd' in asm}}
  __asm__ volatile("subi r30, %0" : : "G" (1)); // expected-error {{value '1' out of range for constraint 'G'}}
}
