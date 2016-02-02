// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

float __attribute__((overloadable)) acos(float); // expected-no-diagnostics
