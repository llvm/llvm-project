// RUN: %clang_cc1 -std=c++1z -verify %s

void x() {
  inline int f(int); // expected-error {{inline declaration of 'f' not allowed in block scope}}
  inline int n; // expected-error {{inline declaration of 'n' not allowed in block scope}}
  static inline int m; // expected-error {{inline declaration of 'm' not allowed in block scope}}
}

inline void g();
struct X {
  inline void f();
  // FIXME: This is ill-formed per [dcl.inline]p5.
  inline void g();
  inline void h() {}
};
