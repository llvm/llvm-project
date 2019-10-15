// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t
// RUN: %clang_cc1 -std=c++2a -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template<typename... T>
concept C = true;

namespace n {
  template<typename... T>
  concept C = true;
}

void f() {
  (void)C<int>;
  (void)C<int, void>;
  (void)n::C<void>;
}

#else /*included pch*/

int main() {
  (void)C<int>;
  (void)C<int, void>;
  (void)n::C<void>;
  f();
}

#endif // HEADER
