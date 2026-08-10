// RUN: %clang_cc1 -verify -emit-llvm-only %s

// Just verify that we don't crash until we support _Imaginary.
double _Imaginary foo; // expected-error {{imaginary types are not supported}}
