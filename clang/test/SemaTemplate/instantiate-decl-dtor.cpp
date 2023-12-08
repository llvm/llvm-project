// RUN: %clang_cc1 %s -fsyntax-only -verify

template <typename T> struct A {
  T x;
  A(int y) { x = y; }
  ~A() { *x = 10; } // expected-error {{indirection requires pointer operand}}
};

void a() {
  A<int> b = 10; // expected-note {{requested here}}
}
