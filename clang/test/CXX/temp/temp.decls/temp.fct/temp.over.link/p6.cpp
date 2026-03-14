// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int N, int M>
struct A0 {
  void g0();
};

template<int X, int Y> void f0(A0<X, Y>) { } // expected-note{{previous}}
template<int N, int M> void f0(A0<M, N>) { }
template<int V1, int V2> void f0(A0<V1, V2>) { } // expected-error{{redefinition}}

template<int X, int Y> void f1(A0<0, (X + Y)>) { } // expected-note{{previous}}
template<int X, int Y> void f1(A0<0, (X - Y)>) { }
template<int A, int B> void f1(A0<0, (A + B)>) { } // expected-error{{redefinition}}

template<int X, int Y> void A0<X, Y>::g0() { }
