// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -triple=i386-pc-win32 %s
// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -triple=i386-pc-win32 -Wformat-non-iso -DNON_ISO_WARNING %s

#include <stdint.h>
int printf(const char *format, ...) __attribute__((format(printf, 1, 2)));
int scanf(const char * restrict, ...) ;
typedef unsigned short wchar_t;

#ifdef NON_ISO_WARNING

// Split off this test to reduce the warning noise in the rest of the file.
void non_iso_warning_test(__int32 i32, __int64 i64, wchar_t c, void *p) {
  printf("%Id", i32); // expected-warning{{'I' length modifier is not supported by ISO C}}
  printf("%I32d", i32); // expected-warning{{'I32' length modifier is not supported by ISO C}}
  printf("%I64d", i64); // expected-warning{{'I64' length modifier is not supported by ISO C}}
  printf("%wc", c); // expected-warning{{'w' length modifier is not supported by ISO C}}
  printf("%Z", p); // expected-warning{{'Z' conversion specifier is not supported by ISO C}}
}

#else

void signed_test(void) {
  short val = 30;
  printf("val = %I64d\n", val); // expected-warning{{format specifies type '__int64' (aka 'long long') but the argument has type 'short'}}
  long long bigval = 30;
  printf("val = %I32d\n", bigval); // expected-warning{{format specifies type '__int32' (aka 'int') but the argument has type 'long long'}}
  printf("val = %Id\n", bigval); // expected-warning{{format specifies type '__int32' (aka 'int') but the argument has type 'long long'}}
}

void unsigned_test(void) {
  unsigned short val = 30;
  printf("val = %I64u\n", val); // expected-warning{{format specifies type 'unsigned __int64' (aka 'unsigned long long') but the argument has type 'unsigned short'}}
  unsigned long long bigval = 30;
  printf("val = %I32u\n", bigval); // expected-warning{{format specifies type 'unsigned __int32' (aka 'unsigned int') but the argument has type 'unsigned long long'}}
  printf("val = %Iu\n", bigval); // expected-warning{{format specifies type 'unsigned __int32' (aka 'unsigned int') but the argument has type 'unsigned long long'}}
}

void w_test(wchar_t c, wchar_t *s) {
  printf("%wc", c);
  printf("%wC", c);
  printf("%C", c);
  printf("%ws", s);
  printf("%wS", s);
  printf("%S", s);
  scanf("%wc", &c);
  scanf("%wC", &c);
  scanf("%C", &c);
  scanf("%ws", s);
  scanf("%wS", s);
  scanf("%S", s);

  double bad;
  printf("%wc", bad); // expected-warning{{format specifies type 'wint_t' (aka 'unsigned short') but the argument has type 'double'}}
  printf("%wC", bad); // expected-warning{{format specifies type 'wchar_t' (aka 'unsigned short') but the argument has type 'double'}}
  printf("%C", bad); // expected-warning{{format specifies type 'wchar_t' (aka 'unsigned short') but the argument has type 'double'}}
  printf("%ws", bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double'}}
  printf("%wS", bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double'}}
  printf("%S", bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double'}}
  scanf("%wc", &bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double *'}}
  scanf("%wC", &bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double *'}}
  scanf("%C", &bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double *'}}
  scanf("%ws", &bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double *'}}
  scanf("%wS", &bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double *'}}
  scanf("%S", &bad); // expected-warning{{format specifies type 'wchar_t *' (aka 'unsigned short *') but the argument has type 'double *'}}

}

void h_test(char c, char* s) {
  double bad;
  printf("%hc", bad); // expected-warning{{format specifies type 'int' but the argument has type 'double'}}
  printf("%hC", bad); // expected-warning{{format specifies type 'int' but the argument has type 'double'}}
  printf("%hs", bad); // expected-warning{{format specifies type 'char *' but the argument has type 'double'}}
  printf("%hS", bad); // expected-warning{{format specifies type 'char *' but the argument has type 'double'}}
  scanf("%hc", &bad); // expected-warning{{format specifies type 'char *' but the argument has type 'double *'}}
  scanf("%hC", &bad); // expected-warning{{format specifies type 'char *' but the argument has type 'double *'}}
  scanf("%hs", &bad); // expected-warning{{format specifies type 'char *' but the argument has type 'double *'}}
  scanf("%hS", &bad); // expected-warning{{format specifies type 'char *' but the argument has type 'double *'}}
}

void z_test(void *p) {
  printf("%Z", p);
  printf("%hZ", p);
  printf("%lZ", p);
  printf("%wZ", p);
  printf("%hhZ", p); // expected-warning{{length modifier 'hh' results in undefined behavior or no effect with 'Z' conversion specifier}}
  scanf("%Z", p); // expected-warning{{invalid conversion specifier 'Z'}}
}

void w_int_test(void) {
  int8_t a;
  int16_t b;
  uint32_t c;
  int64_t d;

  // for %w
  printf("%w8b", a); // expected-warning{{format specifies type 'char' but the argument has type 'int8_t' (aka 'signed char')}}
  printf("%w16i", b);
  printf("%w32u", c);
  printf("%w64x", d);
  scanf("%w8b", a); // expected-warning{{format specifies type 'char' but the argument has type 'int'}}
  scanf("%w16i", b); // expected-warning{{format specifies type 'short' but the argument has type 'int'}}
  scanf("%w32u", c);
  scanf("%w64x", d);

  // unsupported size
  printf("%w92d", a); // expected-warning{{format specifies w width integer type with invalid bit-width 92}}
  scanf("%w0i", b); // expected-warning{{format specifies w width integer type with invalid bit-width 0}}
}

void wf_test(void) {
  int_fast8_t a;
  uint_fast16_t b;
  int_fast32_t c;
  int_fast64_t d;

  // for %wf
  printf("%wf8b", a); // expected-warning{{format specifies type 'char' but the argument has type 'int_fast8_t' (aka 'signed char')}}
  printf("%wf16u", b);
  printf("%wf32o", c);
  printf("%wf64X", d);
  scanf("%wf8b", a); // expected-warning{{format specifies type 'char' but the argument has type 'int'}}
  scanf("%wf16u", b);
  scanf("%wf32o", c);
  scanf("%wf64X", d);

  // unsupported size
  printf("%wf0d", a); // expected-warning{{format specifies wf width integer type with invalid bit-width 0}}
  scanf("%wf35u", b); // expected-warning{{format specifies wf width integer type with invalid bit-width 35}}
}

#endif
