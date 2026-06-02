// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fsyntax-only -verify -Wformat %s

int printf(const char *restrict, ...);
int scanf(const char *restrict, ...);

void t1(float f, double d, long double ld) {
  printf("%Hf", f);
  printf("%He", f);
  printf("%Hg", f);
  printf("%Ha", f);
  printf("%HF", f);
  printf("%HE", f);
  printf("%HG", f);
  printf("%HA", f);

  printf("%Df", d);
  printf("%De", d);
  printf("%Dg", d);
  printf("%Da", d);
  printf("%DF", d);
  printf("%DE", d);
  printf("%DG", d);
  printf("%DA", d);

  printf("%DDf", ld);
  printf("%DDe", ld);
  printf("%DDg", ld);
  printf("%DDa", ld);
  printf("%DDF", ld);
  printf("%DDE", ld);
  printf("%DDG", ld);
  printf("%DDA", ld);
}

void t2(int i, float f, double d) {
  printf("%Df", i);  // expected-warning{{format specifies type '_Decimal64'}}
  printf("%Hf", i);  // expected-warning{{format specifies type '_Decimal32'}}
  printf("%DDf", i); // expected-warning{{format specifies type '_Decimal128'}}
}

void t3(float f) {
  printf("%Hd", f);     // expected-warning{{length modifier 'H' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%Dd", f);     // expected-warning{{length modifier 'D' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%DDd", f);    // expected-warning{{length modifier 'DD' results in undefined behavior or no effect with 'd' conversion specifier}}
  printf("%Hs", "str"); // expected-warning{{length modifier 'H' results in undefined behavior or no effect with 's' conversion specifier}}
}

void t4(double *d_ptr, float *f_ptr, long double *ld_ptr) {
  scanf("%Hf", f_ptr);
  scanf("%Df", d_ptr);
  scanf("%DDf", ld_ptr);
}

void t5(int *i_ptr) {
  scanf("%Df", i_ptr); // expected-warning{{format specifies type '_Decimal64 *'}}
  scanf("%Hf", i_ptr); // expected-warning{{format specifies type '_Decimal32 *'}}
}
