// RUN: %clang_cc1 -std=c++20 %s -fsyntax-only -verify
export struct Unit { // expected-error {{export declaration can only be used within a module interface}}
  bool operator<(const Unit &);
};
