// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify=cxx03 -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify -std=c++11 %s

int main() {
  []{}; // cxx03-warning {{lambdas are a C++11 extension}}
  // expected-no-diagnostics
}
