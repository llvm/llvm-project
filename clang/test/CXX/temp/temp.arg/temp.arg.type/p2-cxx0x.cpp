// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// expected-no-diagnostics

// C++03 imposed restrictions in this paragraph that were lifted with 0x, so we
// just test that the example given now parses cleanly.

template <class T> class X { };
template <class T> void f(T t) { }
struct { } unnamed_obj;
void f() {
  struct A { };
  enum { e1 };
  typedef struct { } B;
  B b;
  X<A> x1;
  X<A*> x2;
  X<B> x3;
  f(e1);
  f(unnamed_obj);
  f(b);
}
