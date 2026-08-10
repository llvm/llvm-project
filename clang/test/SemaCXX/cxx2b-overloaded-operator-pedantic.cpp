// RUN: %clang_cc1 -verify -std=c++23 -pedantic %s
// RUN: %clang_cc1 -verify=compat -std=c++23 -Wpre-c++23-compat %s

// expected-no-diagnostics

struct GH61582 {
  // We accidentally would issue this diagnostic in pedantic mode; show that we
  // only issue it when enabling the compat warnings now.
  void operator[](int, int); // compat-warning {{overloaded 'operator[]' with more than one parameter is a C++23 extension}}
};

