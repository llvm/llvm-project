// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Demonstrate that we don't consider use of 'std' (potentially followed by
// zero or more digits) to be a reserved identifier if it is not the only part
// of the path.
export module std12Three;
