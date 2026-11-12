// RUN: %clang_cc1 -std=c++17 -verify %s -Werror -Wdeprecated -Wno-error=deprecated-redundant-constexpr-static-def

namespace {
  struct A {
    static constexpr int n = 0;
    static constexpr int m = 0;
  };
  constexpr int A::n; // expected-warning{{out-of-line definition of constexpr static data member is redundant in C++17 and is deprecated}}
  const int A::m; // expected-warning{{out-of-line definition of constexpr static data member is redundant in C++17 and is deprecated}}
}
