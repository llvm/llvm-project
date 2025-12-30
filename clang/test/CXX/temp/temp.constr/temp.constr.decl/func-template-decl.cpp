// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

namespace nodiag {

template <typename T> requires (bool(T()))
int A();
template <typename U> requires (bool(U()))
int A();

} // end namespace nodiag

namespace diag {

namespace orig {
  template <typename T> requires true
  int A();
  template <typename T>
  int B();
  template <typename T> requires true
  int C();
}

template <typename T>
int orig::A();
// expected-error@-1{{out-of-line declaration of 'A' does not match any declaration in namespace 'diag::orig'}}
template <typename T> requires true
int orig::B();
// expected-error@-1{{out-of-line declaration of 'B' does not match any declaration in namespace 'diag::orig'}}
template <typename T> requires (!0)
int orig::C();
// expected-error@-1{{out-of-line declaration of 'C' does not match any declaration in namespace 'diag::orig'}}

} // end namespace diag

namespace nodiag {

struct AA {
  template <typename T> requires (someFunc(T()))
  int A();
};

template <typename T> requires (someFunc(T()))
int AA::A() { return sizeof(T); }

} // end namespace nodiag

namespace diag {

template <unsigned N>
struct TA { // #defined-here
  template <template <unsigned> class TT> requires TT<N>::happy
  int A();
};

template <unsigned N>
template <template <unsigned> class TT> int TA<N>::A() { return sizeof(TT<N>); }
// expected-error@-1{{out-of-line definition of 'A' does not match any declaration in 'diag::TA<N>'}}
// expected-note@#defined-here{{defined here}}

} // end namespace diag
