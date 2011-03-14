// RUN: true
// Disabled for 2.9
//%clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct B1 {
  B1(int);
  B1(int, int);
};
struct D1 : B1 {
  using B1::B1;
};
D1 d1a(1), d1b(1, 1);

D1 fd1() { return 1; }

struct B2 {
  explicit B2(int, int = 0, int = 0);
};
struct D2 : B2 { // expected-note {{candidate constructor}}
  using B2::B2;
};
D2 d2a(1), d2b(1, 1), d2c(1, 1, 1);

D2 fd2() { return 1; } // expected-error {{no viable conversion}}

struct B3 {
  B3(void*); // expected-note {{inherited from here}}
};
struct D3 : B3 { // expected-note {{candidate constructor}}
  using B3::B3; // expected-note {{candidate constructor (inherited)}}
};
D3 fd3() { return 1; } // expected-error {{no viable conversion}}
