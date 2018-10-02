// RUN: %clang_cc1 %s -verify -Wconversion -fsyntax-only -triple x86_64-pc-linux-gnu
// RUN: %clang_cc1 %s -verify -Wconversion -fsyntax-only -triple i386-pc-linux-gnu

void test(void) {
  unsigned int a = 1;

  unsigned long long b = -a; // expected-warning {{higher order bits are zeroes after implicit conversion}}
  long long c = -a;          // expected-warning {{the resulting value is always non-negative after implicit conversion}}

  unsigned long b2 = -a;
#ifdef __x86_64__
// expected-warning@-2 {{higher order bits are zeroes after implicit conversion}}
#endif
  long c2 = -a;
#ifdef __x86_64__
// expected-warning@-2 {{the resulting value is always non-negative after implicit conversion}}
#else
// expected-warning@-4 {{implicit conversion changes signedness: 'unsigned int' to 'long'}}
#endif
}
