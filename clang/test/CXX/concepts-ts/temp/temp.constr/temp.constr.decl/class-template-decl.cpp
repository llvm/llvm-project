// RUN: %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s

namespace nodiag {

template <typename T> requires bool(T())
struct A;
template <typename U> requires bool(U())
struct A;

} // end namespace nodiag

namespace diag {

template <typename T> requires true // expected-note{{previous template declaration is here}}
struct A;
template <typename T> struct A; // expected-error{{associated constraints differ in template redeclaration}}

template <typename T> struct B; // expected-note{{previous template declaration is here}}
template <typename T> requires true // expected-error{{associated constraints differ in template redeclaration}}
struct B;

template <typename T> requires true // expected-note{{previous template declaration is here}}
struct C;
template <typename T> requires !0 // expected-error{{associated constraints differ in template redeclaration}}
struct C;

} // end namespace diag

namespace nodiag {

struct AA {
  template <typename T> requires someFunc(T())
  struct A;
};

template <typename T> requires someFunc(T())
struct AA::A { };

struct AAF {
  template <typename T> requires someFunc(T())
  friend struct AA::A;
};

} // end namespace nodiag

namespace diag {

template <unsigned N>
struct TA {
  template <template <unsigned> class TT> requires TT<N>::happy // expected-note 2{{previous template declaration is here}}
  struct A;

  struct AF;
};

template <unsigned N>
template <template <unsigned> class TT> struct TA<N>::A { }; // expected-error{{associated constraints differ in template redeclaration}}

template <unsigned N>
struct TA<N>::AF {
  template <template <unsigned> class TT> requires TT<N + 0>::happy // expected-error{{associated constraints differ in template redeclaration}}
  friend struct TA::A;
};

} // end namespace diag
