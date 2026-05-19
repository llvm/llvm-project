// RUN: %clang_cc1 -std=c++26 -Wno-invalid-pp-token %s -fsyntax-only -verify

// [cpp.concat]/p3: If the result is not a valid
// preprocessing token, the program is ill-formed.
#define CONCAT(A, B) A ## B
CONCAT(=, >) // expected-error {{pasting formed '=>', an invalid preprocessing token}}
// expected-error@-1 {{expected unqualified-id}}
