// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

template <typename>
class Bar {
  Bar<int> *variables_to_modify;
  foo() { // expected-error {{C++ requires a type specifier for all declarations}}
    for (auto *c : *variables_to_modify)
      delete c;
  }
};
