// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify -std=c++03 %s

void func() {
  []() {}; // expected-warning {{lambdas are a C++11 extension}}
}
