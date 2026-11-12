// RUN: %clang_cc1 -fsyntax-only -verify %s

__attribute__((common)) int x; // expected-error {{'common' attribute is not supported in C++}}
