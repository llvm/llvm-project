// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fsyntax-only -verify %s

int printf(const char *restrict, ...);

struct bitfields {
  long a : 2;
  unsigned long b : 2;
  long c : 32;          // assumes that int is 32 bits
  unsigned long d : 32; // assumes that int is 32 bits
} bf;

void bitfield_promotion(void) {
  printf("%ld", bf.a); // expected-warning {{format specifies type 'long' but the argument has type 'int'}}
  printf("%lu", bf.b); // expected-warning {{format specifies type 'unsigned long' but the argument has type 'int'}}
  printf("%ld", bf.c); // expected-warning {{format specifies type 'long' but the argument has type 'int'}}
  printf("%lu", bf.d); // expected-warning {{format specifies type 'unsigned long' but the argument has type 'unsigned int'}}
}
