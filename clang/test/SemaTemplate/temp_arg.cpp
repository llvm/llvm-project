// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 %std_cxx17- %s
template<typename T, 
         int I, 
         template<typename> class TT>
  class A; // precxx17-note 3 {{template is declared here}} \
              cxx17-note 2 {{template is declared here}} \
              cxx17-note {{candidate template ignored: couldn't infer template argument 'T'}} \
              cxx17-note {{candidate function template not viable: requires 1 argument, but 0 were provided}}

template<typename> class X;

A<int, 0, X> * a1;

A<float, 1, X, double> *a2; // expected-error{{too many template arguments for class template 'A'}}
A<float, 1> *a3; // expected-error{{too few template arguments for class template 'A'}}
A a4; // precxx17-error{{use of class template 'A' requires template arguments}} \
         cxx17-error{{no viable constructor or deduction guide for deduction of template arguments of 'A'}}

namespace test0 {
  template <class t> class foo {};
  template <class t> class bar {
    bar(::test0::foo<tee> *ptr) {} // expected-error {{use of undeclared identifier 'tee'}}
  };
}
