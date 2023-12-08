// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

typedef __attribute__((ext_vector_type(16))) _Bool bool8; // expected-error{{invalid vector element type 'bool'}}
