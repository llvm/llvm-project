// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ %s -std=c++11 -fsyntax-only -verify
// RUN: not %clang_cc1 -x c++ %t -std=c++11 -fixit
// RUN: %clang_cc1 -x c++ %t -std=c++11 -fsyntax-only

constexpr int main() { } // expected-error{{'main' is not allowed to be declared constexpr}}
