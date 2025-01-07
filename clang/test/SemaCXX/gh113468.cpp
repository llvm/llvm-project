// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

constexpr int expr() {
  if (({
        int f;
        f = 0;
        if (f)
          break; // expected-error {{'break' statement not in loop or switch statement}}
      }))
    return 2;
  return 1;
}
