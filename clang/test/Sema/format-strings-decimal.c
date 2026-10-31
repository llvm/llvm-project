// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fsyntax-only -verify -Wformat %s

int printf(const char *restrict, ...);
int scanf(const char *restrict, ...);

void t1(float f, double d, long double ld) {
  printf("%Hf", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%He", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%Hg", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%Ha", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%HF", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%HE", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%HG", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%HA", f);   // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}

  printf("%Df", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%De", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%Dg", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%Da", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%DF", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%DE", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%DG", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%DA", d);   // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}

  printf("%DDf", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
  printf("%DDe", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
  printf("%DDg", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
  printf("%DDa", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
  printf("%DDF", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
  printf("%DDE", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
  printf("%DDG", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
  printf("%DDA", ld); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
}

void t2(int i) {
  printf("%Df", i);  // expected-warning{{format specifies type '_Decimal64' which is not supported yet}}
  printf("%Hf", i);  // expected-warning{{format specifies type '_Decimal32' which is not supported yet}}
  printf("%DDf", i); // expected-warning{{format specifies type '_Decimal128' which is not supported yet}}
}

void t3(float f) {
  printf("%Hd", f);     // expected-warning{{length modifier 'H' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%Dd", f);     // expected-warning{{length modifier 'D' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%DDd", f);    // expected-warning{{length modifier 'DD' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%Hs", "str"); // expected-warning{{length modifier 'H' results in undefined behavior or no effect with 's' conversion specifier}}
}

void t4(double *d_ptr, float *f_ptr, long double *ld_ptr) {
  scanf("%Hf", f_ptr);   // expected-warning{{format specifies type '_Decimal32 *' which is not supported yet}}
  scanf("%Df", d_ptr);   // expected-warning{{format specifies type '_Decimal64 *' which is not supported yet}}
  scanf("%DDf", ld_ptr); // expected-warning{{format specifies type '_Decimal128 *' which is not supported yet}}
}

void t5(int *i_ptr) {
  scanf("%Df", i_ptr); // expected-warning{{format specifies type '_Decimal64 *' which is not supported yet}}
  scanf("%Hf", i_ptr); // expected-warning{{format specifies type '_Decimal32 *' which is not supported yet}}
}
