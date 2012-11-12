// RUN: %clang_cc1 -fsyntax-only -verify -triple i386-apple-darwin9 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple thumbv6-apple-ios4.0 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-mingw32 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-pc-win32 %s

// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-gnu -DALLOWED %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-freebsd -DALLOWED %s

int printf(const char *restrict, ...);
int scanf(const char * restrict, ...) ;

void test() {
  long notLongEnough = 1;
  long long quiteLong = 2;

  printf("%Ld", notLongEnough); // expected-warning {{format specifies type 'long long' but the argument has type 'long'}}
  printf("%Ld", quiteLong);

#ifndef ALLOWED
  // expected-warning@-4 {{length modifier 'L' results in undefined behavior or no effect with 'd' conversion specifier}}
  // expected-note@-5 {{did you mean to use 'll'?}}

  // expected-warning@-6 {{length modifier 'L' results in undefined behavior or no effect with 'd' conversion specifier}}
  // expected-note@-7 {{did you mean to use 'll'?}}
#endif
}

void testAlwaysInvalid() {
  // We should not suggest 'll' here!
  printf("%Lc", 'a'); // expected-warning {{length modifier 'L' results in undefined behavior or no effect with 'c' conversion specifier}}
  printf("%Ls", "a"); // expected-warning {{length modifier 'L' results in undefined behavior or no effect with 's' conversion specifier}}
}

#ifdef ALLOWED
// PR 9466: clang: doesn't know about %Lu, %Ld, and %Lx
void printf_longlong(long long x, unsigned long long y) {
  printf("%Ld", y); // no-warning
  printf("%Lu", y); // no-warning
  printf("%Lx", y); // no-warning
  printf("%Ld", x); // no-warning
  printf("%Lu", x); // no-warning
  printf("%Lx", x); // no-warning
  printf("%Ls", "hello"); // expected-warning {{length modifier 'L' results in undefined behavior or no effect with 's' conversion specifier}}
}

void scanf_longlong(long long *x, unsigned long long *y) {
  scanf("%Ld", y); // no-warning
  scanf("%Lu", y); // no-warning
  scanf("%Lx", y); // no-warning
  scanf("%Ld", x); // no-warning
  scanf("%Lu", x); // no-warning
  scanf("%Lx", x); // no-warning
  scanf("%Ls", "hello"); // expected-warning {{length modifier 'L' results in undefined behavior or no effect with 's' conversion specifier}}
}
#endif
