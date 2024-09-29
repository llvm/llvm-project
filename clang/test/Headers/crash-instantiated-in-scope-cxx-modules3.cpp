// RUN: %clang_cc1 %s -std=c++11 -emit-pch -o %t
// RUN: %clang_cc1 %s -std=c++11 -include-pch %t -fsyntax-only -verify

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// No crash or assertion failure on multiple nested lambdas deserialization.
template <typename T>
void b() {
  [] {
    []{
      []{
        []{
          []{
          }();
        }();
      }();
    }();
  }();
}

void foo() {
  b<int>();
}
#endif
