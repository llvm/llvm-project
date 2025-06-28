// RUN: %clang_cc1 -fsyntax-only -verify %s

void test(){
    (void) atoll;// expected-error {{use of undeclared identifier}} expected-note {{maybe try}} expected-note {{'atoll' is a}}
}
