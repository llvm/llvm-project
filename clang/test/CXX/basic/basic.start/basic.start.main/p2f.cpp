// RUN: %clang_cc1 -fsyntax-only -verify %s 

void  // expected-error {{'main' must return 'int'}}
main( // expected-error {{first parameter of 'main' (argument count) must be of type 'int'}}
     float a
) {
}
