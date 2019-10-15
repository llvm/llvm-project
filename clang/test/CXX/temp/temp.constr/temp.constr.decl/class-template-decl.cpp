// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

namespace nodiag {

template <typename T> requires bool(T())
struct A;
template <typename U> requires bool(U())
struct A;

} // end namespace nodiag

namespace diag {

template <typename T> requires true // expected-note{{previous template declaration is here}}
struct A;
template <typename T> struct A; // expected-error{{requires clause differs in template redeclaration}}

template <typename T> struct B; // expected-note{{previous template declaration is here}}
template <typename T> requires true // expected-error{{requires clause differs in template redeclaration}}
struct B;

template <typename T> requires true // expected-note{{previous template declaration is here}}
struct C;
template <typename T> requires !0 // expected-error{{requires clause differs in template redeclaration}}
struct C;

} // end namespace diag

namespace nodiag {

struct AA {
  template <typename T> requires someFunc(T())
  struct A;
};

template <typename U> requires someFunc(U())
struct AA::A { };

struct AAF {
  template <typename T> requires someFunc(T())
  friend struct AA::A;
};

} // end namespace nodiag

namespace diag {

template <unsigned N>
struct TA {
  template <template <unsigned> class TT> requires TT<N>::happy // expected-note {{previous template declaration is here}}
  struct A;

  template <template <unsigned> class TT> requires TT<N>::happy // expected-note {{previous template declaration is here}}
  struct B;

  struct AF;
};

template <unsigned N>
template <template <unsigned> class TT> struct TA<N>::A { }; // expected-error{{requires clause differs in template redeclaration}}


template <unsigned N>
template <template <unsigned> class TT> requires TT<N + 1>::happy struct TA<N>::B { }; // expected-error{{requires clause differs in template redeclaration}}

template <unsigned N>
struct TA<N>::AF {
  // we do not expect a diagnostic here because the template parameter list is dependent.
  template <template <unsigned> class TT> requires TT<N + 0>::happy
  friend struct TA::A;
};

} // end namespace diag
