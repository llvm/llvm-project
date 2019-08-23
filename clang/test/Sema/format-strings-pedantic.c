// RUN: %clang_cc1 -fsyntax-only -verify -Wformat -Wformat-pedantic -isystem %S/Inputs %s

int printf(const char *restrict, ...);

typedef unsigned char uint8_t;

void print_char_as_short() {
  printf("%hu\n", (unsigned char)1); // expected-warning{{format specifies type 'unsigned short' but the argument has type 'unsigned char'}}
  printf("%hu\n", (uint8_t)1);       // expected-warning{{format specifies type 'unsigned short' but the argument has type 'uint8_t' (aka 'unsigned char')}}
}
