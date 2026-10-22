// RUN: %clang_cc1 -fsyntax-only %s -verify

a[i] = b[i]; // expected-error {{use of undeclared identifier 'i'}} \
                expected-error {{a type specifier is required for all declarations}} \
                expected-error {{use of undeclared identifier 'b'}} \
                expected-error {{use of undeclared identifier 'i'}}
extern char b[];
extern char a[];

void foo(int j) {
  a[j] = b[j];
}
