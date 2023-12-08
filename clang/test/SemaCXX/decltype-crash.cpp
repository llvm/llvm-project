// RUN: %clang_cc1 -fsyntax-only -verify -Wc++11-compat -std=c++98 %s

int& a();

void f() {
  decltype(a()) c; // expected-warning {{'decltype' is a keyword in C++11}} \
                   // expected-error {{use of undeclared identifier 'decltype'}} \
                   // expected-error {{expected ';' after expression}} \
                   // expected-error {{use of undeclared identifier 'c'}}
}
