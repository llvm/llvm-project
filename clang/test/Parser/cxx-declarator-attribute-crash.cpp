// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-error@+5{{brackets are not allowed here}}
// expected-error@+4{{a type specifier is required for all declarations}}
// expected-warning@+3{{unknown attribute 'h' ignored}}
// expected-error@+2{{definition of variable with array type}}
// expected-error@+1{{expected ';'}}
[][[h]]l
