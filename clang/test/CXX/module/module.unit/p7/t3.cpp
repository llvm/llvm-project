// This tests whether the global module would be created when the program don't declare it explicitly.
// RUN: %clang_cc1 -std=c++20 %s -verify
// expected-no-diagnostics
export module x;

extern "C" void foo();
extern "C++" class CPP {};
