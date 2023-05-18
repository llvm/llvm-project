// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fsyntax-only -verify %s

// In C++, the bitfield promotion from long to int does not occur, unlike C.
// expected-no-diagnostics

int printf(const char *restrict, ...);

struct bitfields {
  long a : 2;
  unsigned long b : 2;
  long c : 32;          // assumes that int is 32 bits
  unsigned long d : 32; // assumes that int is 32 bits
} bf;

void bitfield_promotion() {
  printf("%ld", bf.a);
  printf("%lu", bf.b);
  printf("%ld", bf.c);
  printf("%lu", bf.d);
}
