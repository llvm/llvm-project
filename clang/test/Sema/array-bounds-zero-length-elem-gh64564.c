// RUN: %clang_cc1 -triple i686-apple-darwin -verify %s

int a[][0]; // expected-warning {{tentative array definition assumed to have one element}}
void gh64564_1(void) {
  int b = a[0x100000000][0];
}

typedef struct {} S;
S s[]; // expected-warning {{tentative array definition assumed to have one element}}
void gh64564_2(void) {
  S t = s[0x100000000];
}
