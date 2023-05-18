// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T> struct A; // expected-note 4{{template is declared here}}

template<typename T> struct B : A<T*> { }; // expected-error{{implicit instantiation of undefined template}} \
// expected-error{{implicit instantiation of undefined template 'A<X *>'}}

template<typename T> struct C : B<T> { } ; // expected-note{{instantiation of template class}}

template<typename T> struct D : C<T> { }; // expected-note{{instantiation of template class}}

template<typename T> struct E : D<T> { }; // expected-note{{instantiation of template class}}

template<typename T> struct F : E<T(T)> { }; // expected-note{{instantiation of template class}}

void f() {
 (void)sizeof(F<int>); // expected-note{{instantiation of template class}}
}

typedef struct { } X;

void g() {
  (void)sizeof(B<X>); // expected-note{{in instantiation of template class 'B<X>' requested here}}
}

template<typename T> 
struct G : A<T>, // expected-error{{implicit instantiation of undefined template 'A<int>'}}
  A<T*> // expected-error{{implicit instantiation of undefined template 'A<int *>'}}
  { };

void h() {
  (void)sizeof(G<int>); // expected-note{{in instantiation of template class 'G<int>' requested here}}
}

namespace PR13365 {
  template <class T> class ResultTy { // expected-warning {{does not declare any constructor}}
    T t; // expected-note {{reference member 't' will never be initialized}}
  };

  template <class T1, class T2>
    typename ResultTy<T2>::error Deduce( void (T1::*member)(T2) ) {} // \
    // expected-note {{instantiation of template class 'PR13365::ResultTy<int &>'}} \
    // expected-note {{substitution failure [with T1 = PR13365::Cls, T2 = int &]}}

  struct Cls {
    void method(int&);
  };
  void test() {
    Deduce(&Cls::method); // expected-error {{no matching function}} \
                          // expected-note {{substituting deduced template arguments into function template 'Deduce' [with T1 = PR13365::Cls, T2 = int &]}}
  }
}
