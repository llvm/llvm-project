// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

namespace ex1 {
  template<int i> class A { };
  template<short s> void f(A<s>);
  // expected-note@-1 {{candidate template ignored: substitution failure: deduced non-type template argument does not have the same type as the corresponding template parameter ('int' vs 'short')}}
  void k1() {
    A<1> a;
    f(a); // expected-error {{no matching function for call to 'f'}}
    f<1>(a);
  }
}

namespace ex2 {
  template<const short cs> class B { };
  template<short s> void g(B<s>);
  void k2() {
    B<1> b;
    g(b);
  }
}

#if __cplusplus >= 201703L
namespace ex3 {
  template<auto> struct C;
  template<long long x> void f(C<x> *);
  void g(C<0LL> *ap) { f(ap); }
}

namespace ex4 {
  template<int> struct D;
  template<auto x> void f(D<x> *);
  void g(D<0LL> *ap) { f(ap); }
}

namespace ex5 {
  template<int &> struct E;
  template<auto x> void f(E<x> *);
  // expected-note@-1 {{candidate template ignored: substitution failure: non-type template argument is not a constant expression}}
  int v;
  void g(E<v> *bp) { f(bp); } // expected-error {{no matching function for call to 'f'}}
}

namespace ex6 {
  template<const int &> struct F;
  template<decltype(auto) x> void f(F<x> *);
  int i;
  void g(F<i> *ap) { f(ap); }
}

namespace ex7 {
  template <decltype(auto) q> struct G;
  template <auto x> long *f(G<x> *);
  template <decltype(auto) x> short *f(G<x> *);
  const int j = 0;
  short *g1(G<(j)> *ap) { return f(ap); }
  long *g2(G<j> *ap) { return f(ap); }
}
#endif
