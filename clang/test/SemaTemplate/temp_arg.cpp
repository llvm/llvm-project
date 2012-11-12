// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T, 
         int I, 
         template<typename> class TT>
  class A; // expected-note 3 {{template is declared here}}

template<typename> class X;

A<int, 0, X> * a1;

A<float, 1, X, double> *a2; // expected-error{{too many template arguments for class template 'A'}}
A<float, 1> *a3; // expected-error{{too few template arguments for class template 'A'}}
A a3; // expected-error{{use of class template A requires template arguments}}

namespace test0 {
  template <class t> class foo {};
  template <class t> class bar {
    bar(::test0::foo<tee> *ptr) {} // FIXME(redundant): expected-error 2 {{use of undeclared identifier 'tee'}}
  };
}
