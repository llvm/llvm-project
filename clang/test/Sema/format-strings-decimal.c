// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fsyntax-only -verify -Wformat %s

int printf(const char *restrict, ...);
int scanf(const char *restrict, ...);

void t1(float f, double d, long double ld) {
  printf("%Hf", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%He", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%Hg", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%Ha", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%HF", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%HE", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%HG", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%HA", f);   // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}

  printf("%Df", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%De", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%Dg", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%Da", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%DF", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%DE", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%DG", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%DA", d);   // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}

  printf("%DDf", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
  printf("%DDe", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
  printf("%DDg", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
  printf("%DDa", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
  printf("%DDF", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
  printf("%DDE", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
  printf("%DDG", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
  printf("%DDA", ld); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
}

void t2(int i) {
  printf("%Df", i);  // expected-warning{{format specifier requires type '_Decimal64' which is not supported}}
  printf("%Hf", i);  // expected-warning{{format specifier requires type '_Decimal32' which is not supported}}
  printf("%DDf", i); // expected-warning{{format specifier requires type '_Decimal128' which is not supported}}
}

void t3(float f) {
  printf("%Hd", f);     // expected-warning{{length modifier 'H' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%Dd", f);     // expected-warning{{length modifier 'D' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%DDd", f);    // expected-warning{{length modifier 'DD' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%Hs", "str"); // expected-warning{{length modifier 'H' results in undefined behavior or no effect with 's' conversion specifier}}
}

void t4(double *d_ptr, float *f_ptr, long double *ld_ptr) {
  scanf("%Hf", f_ptr);   // expected-warning{{format specifier requires type '_Decimal32 *' which is not supported}}
  scanf("%Df", d_ptr);   // expected-warning{{format specifier requires type '_Decimal64 *' which is not supported}}
  scanf("%DDf", ld_ptr); // expected-warning{{format specifier requires type '_Decimal128 *' which is not supported}}
}

void t5(int *i_ptr) {
  scanf("%Df", i_ptr); // expected-warning{{format specifier requires type '_Decimal64 *' which is not supported}}
  scanf("%Hf", i_ptr); // expected-warning{{format specifier requires type '_Decimal32 *' which is not supported}}
}
