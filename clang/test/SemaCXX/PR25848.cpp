// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A;

inline int g();  // expected-warning{{inline function 'g' is not defined}}

template<int M>
struct R {
  friend int g() {
    return M;
  }
};

void m() {
  g();  // expected-note{{used here}}
}
