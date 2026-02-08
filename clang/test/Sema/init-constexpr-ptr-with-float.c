// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

// PR180313: Fix crashes when initializing constexpr int* with a floating-point value

constexpr int *p = 0.0; // expected-error {{initializing 'int *const' with an expression of incompatible type 'double'}}