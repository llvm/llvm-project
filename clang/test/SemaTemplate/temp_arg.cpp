// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 %std_cxx17- %s
template<typename T,
         int I,
         template<typename> class TT> // expected-note {{template parameter is declared here}}
  class A; // precxx17-note 2 {{template is declared here}} \
              cxx17-note {{template is declared here}} \
              cxx17-note {{candidate template ignored: couldn't infer template argument 'T'}} \
              cxx17-note {{implicit deduction guide declared as 'template <typename T, int I, template <typename> class TT> A(A<T, I, TT>) -> A<T, I, TT>'}} \
              cxx17-note {{candidate function template not viable: requires 1 argument, but 0 were provided}} \
              cxx17-note {{implicit deduction guide declared as 'template <typename T, int I, template <typename> class TT> A() -> A<T, I, TT>'}} \

template<typename> class X;

A<int, 0, X> * a1;

A<float, 1, X, double> *a2; // expected-error{{too many template arguments for class template 'A'}}
A<float, 1> *a3; // expected-error{{missing template argument for template parameter}}
A a4; // precxx17-error{{use of class template 'A' requires template arguments}} \
         cxx17-error{{no viable constructor or deduction guide for deduction of template arguments of 'A'}}

namespace test0 {
  template <class t> class foo {};
  template <class t> class bar {
    bar(::test0::foo<tee> *ptr) {} // expected-error {{use of undeclared identifier 'tee'}}
  };
}
