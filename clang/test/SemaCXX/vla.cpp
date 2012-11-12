// RUN: %clang_cc1 -verify %s

// PR11925
int n;
int (&f())[n]; // expected-error {{function declaration cannot have variably modified type}}
