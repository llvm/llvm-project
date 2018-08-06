// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++2a

int foo(int x) {
  return x == x; // expected-warning {{self-comparison always evaluates to true}}
}

struct X {
  bool operator==(const X &x);
};

struct A {
  int x;
  X x2;
  int a[3];
  int b[3];
  bool f() { return x == x; } // expected-warning {{self-comparison always evaluates to true}}
  bool g() { return x2 == x2; } // no-warning
  bool h() { return a == b; } // expected-warning {{array comparison always evaluates to false}}
  bool i() {
    int c[3];
    return a == c; // expected-warning {{array comparison always evaluates to false}}
  }
};

namespace NA { extern "C" int x[3]; }
namespace NB { extern "C" int x[3]; }
bool k = NA::x == NB::x; // expected-warning {{self-comparison always evaluates to true}}

template<typename T> struct Y { static inline int n; };
bool f() {
  return
    Y<int>::n == Y<int>::n || // expected-warning {{self-comparison always evaluates to true}}
    Y<void>::n == Y<int>::n;
}
template<typename T, typename U>
bool g() {
  // FIXME: Ideally we'd produce a self-comparison warning on the first of these.
  return
    Y<T>::n == Y<T>::n ||
    Y<T>::n == Y<U>::n;
}
template bool g<int, int>(); // should not produce any warnings
