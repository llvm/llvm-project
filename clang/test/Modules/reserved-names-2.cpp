// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Demonstrate that we don't consider use of 'std' followed by digits to be a
// reserved identifier if it is not the first part of the path.
export module should_succeed.std0;
