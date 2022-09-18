// RUN: %clang_cc1 -std=c++20 -fmodule-name=test -fsyntax-only %s -verify
// expected-no-diagnostics

// Ensure that we won't crash if we specified `-fmodule-name` in `c++20`
// for a non module unit.
int a;
