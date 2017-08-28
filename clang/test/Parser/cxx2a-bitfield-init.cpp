// RUN: %clang_cc1 -std=c++2a -verify %s

namespace std_example {
  int a;
  const int b = 0; // expected-note {{here}}
  struct S {
    int x1 : 8 = 42;
    int x2 : 8 { 42 };
    int y1 : true ? 8 : a = 42;
    int y3 : (true ? 8 : b) = 42;
    int z : 1 || new int { 1 };
  };
  static_assert(S{}.x1 == 42);
  static_assert(S{}.x2 == 42);
  static_assert(S{}.y1 == 0);
  static_assert(S{}.y3 == 42);
  static_assert(S{}.z == 0);

  struct T {
    int y2 : true ? 8 : b = 42; // expected-error {{cannot assign to variable 'b'}}
  };
}
