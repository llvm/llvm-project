// RUN: %clang_cc1 -x c -verify %s
// RUN: %clang_cc1 -x c++ -verify %s
// expected-no-diagnostics

// precise is a keyword only in HLSL.
int precise;
