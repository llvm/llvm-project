// RUN: %clang_cc1 %s -verify -Wsource-uses-openacc
// expected-warning@+1{{unexpected '#pragma acc ...' in program}}
#pragma acc foo bar baz blitz.
int foo;
