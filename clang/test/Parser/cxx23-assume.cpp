// RUN: %clang_cc1 -std=c++23 -x c++ %s -verify

void f(int x, int y) {
  [[assume(true)]];
  [[assume(1)]];
  [[assume(1.0)]];
  [[assume(1 + 2 == 3)]];
  [[assume(x ? 1 : 2)]];
  [[assume(x && y)]];
  [[assume(true)]] [[assume(true)]];

  [[assume(x = 2)]]; // expected-error {{requires parentheses}}
  [[assume(2, 3)]]; // expected-error {{requires parentheses}}
}
