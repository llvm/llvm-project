// RUN: %clang_cc1 -fsyntax-only -verify %s

void test(){
    (void) atoll;// expected-error {{use of undeclared identifier}} expected-note {{'atoll' is defined in}} expected-note {{'atoll' is a}}
}
