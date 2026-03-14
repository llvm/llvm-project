// RUN: %clang_cc1 -fsyntax-only -Wunused -verify %s

struct X0 {
  ~X0();
};

struct X1 { };

template<typename T>
void f() {
  X0 x0;
  X1 x1; // expected-warning{{unused variable 'x1'}}
}

template<typename T, typename U>
void g() {
  T t;
  U u; // expected-warning{{unused variable 'u'}}
}

template void g<X0, X1>(); // expected-note{{in instantiation of}}
