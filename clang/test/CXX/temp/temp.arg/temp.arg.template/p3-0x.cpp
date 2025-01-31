// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template <class T> struct eval; // expected-note 3{{template is declared here}}

template <template <class, class...> class TT, class T1, class... Rest>
struct eval<TT<T1, Rest...>> { };

template <class T1> struct A;
template <class T1, class T2> struct B;
template <int N> struct C;
template <class T1, int N> struct D;
template <class T1, class T2, int N = 17> struct E;

eval<A<int>> eA;
eval<B<int, float>> eB;
eval<C<17>> eC; // expected-error{{implicit instantiation of undefined template 'eval<C<17>>'}}
eval<D<int, 17>> eD; // expected-error{{implicit instantiation of undefined template 'eval<D<int, 17>>'}}
eval<E<int, float>> eE; // expected-error{{implicit instantiation of undefined template 'eval<E<int, float>>}}

template<
  template <int ...N> // expected-error {{cannot be narrowed from type 'int' to 'short'}}
                      // expected-error@-1 {{conversion from 'int' to 'void *' is not allowed in a converted constant expression}}
  class TT // expected-note 2{{template parameter is declared here}}
> struct X0 { };

template<int I, int J, int ...Rest> struct X0a;
template<int ...Rest> struct X0b;
template<int I, long J> struct X0c;
template<int I, short J> struct X0d; // expected-note {{template parameter is declared here}}
template<int I, void *J> struct X0e; // expected-note {{template parameter is declared here}}

X0<X0a> inst_x0a;
X0<X0b> inst_x0b;
X0<X0c> inst_x0c;
X0<X0d> inst_x0d; // expected-note {{template template argument is incompatible}}
X0<X0e> inst_x0e; // expected-note {{template template argument is incompatible}}

template<typename T,
         template <T ...N> // expected-error {{conversion from 'short' to 'void *' is not allowed in a converted constant expression}}
                           // expected-error@-1 {{cannot be narrowed from type 'int' to 'short'}}
         class TT // expected-note 2{{template parameter is declared here}}
> struct X1 { };

template<int I, int J, int ...Rest> struct X1a;
template<long I, long ...Rest> struct X1b;
template<short I, short J> struct X1c;
template<short I, long J> struct X1d;  // expected-note {{template parameter is declared here}}
template<short I, void *J> struct X1e; // expected-note {{template parameter is declared here}}

X1<int, X1a> inst_x1a;
X1<long, X1b> inst_x1b;
X1<short, X1c> inst_x1c;
X1<short, X1d> inst_sx1d;
X1<int, X1d> inst_ix1d;  // expected-note {{template template argument is incompatible}}
X1<short, X1e> inst_x1e; // expected-note {{template template argument is incompatible}}

template <int> class X2; // expected-note{{template is declared here}} \
                         // expected-note{{template is declared here}}
class X3 : X2<1> {}; // expected-error{{implicit instantiation of undefined template 'X2<1>'}}

template <int> class X4 : X3 {
  struct {
    X2<1> e; // expected-error{{implicit instantiation of undefined template 'X2<1>'}}
  } f;
};
