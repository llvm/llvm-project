// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

int test['\\' == 92 ? 1 : -1];
int test['\"' == 34 ? 1 : -1];
int test['\'' == 39 ? 1 : -1];
int test['\?' == 63 ? 1 : -1];
int test['\a' == 7 ? 1 : -1];
int test['\b' == 8 ? 1 : -1];
int test['\e' == 27 ? 1 : -1]; // expected-warning {{non-standard escape}}
int test['\E' == 27 ? 1 : -1]; // expected-warning {{non-standard escape}}
int test['\f' == 12 ? 1 : -1];
int test['\n' == 10 ? 1 : -1];
int test['\r' == 13 ? 1 : -1];
int test['\t' == 9 ? 1 : -1];
int test['\v' == 11 ? 1 : -1];
int test['\xa' == 10 ? 1 : -1];
int test['\1' == 1 ? 1 : -1];
int test['\(' == 40 ? 1 : -1]; // expected-warning {{non-standard escape}}
int test['\{' == 123 ? 1 : -1]; // expected-warning {{non-standard escape}}
int test['\[' == 91 ? 1 : -1]; // expected-warning {{non-standard escape}}
int test['\%' == 37 ? 1 : -1]; // expected-warning {{non-standard escape}}
const char *format = "abc \m def"; // expected-warning{{unknown escape sequence '\m'}}
