// RUN: %clang_cc1 -triple x86_64-apple-darwin -verify %s

void f() {
  int a[2147483647U][2147483647U]; // expected-error{{array is too large}}
  int b[1073741825U - 1U][2147483647U];
  int c[18446744073709551615U/sizeof(int)/2];
}
