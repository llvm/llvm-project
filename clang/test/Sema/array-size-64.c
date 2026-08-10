// RUN: %clang_cc1 -triple x86_64-apple-darwin -verify %s

void f(void) {
  int a[2147483647U][2147483647U]; // expected-error{{array is too large}}
  int b[1073741825U - 1U][2147483647U]; // expected-error{{array is too large}}
}

void pr8256(void) {
  typedef char a[1LL<<61];  // expected-error {{array is too large}}
  typedef char b[(long long)sizeof(a)-1];
}

