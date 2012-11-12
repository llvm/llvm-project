// RUN: %clang_cc1 -fsyntax-only -verify %s 

static int main() { // expected-error {{'main' is not allowed to be declared static}}
}
