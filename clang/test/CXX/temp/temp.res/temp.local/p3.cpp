// RUN: %clang_cc1 -verify=expected,precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -verify=expected,cxx17 %std_cxx17- %s

template <class T> struct Base {
  // expected-note@-1 2{{member type 'Base<int>' found by ambiguous name lookup}}
  // expected-note@-2 2{{member type 'Base<char>' found by ambiguous name lookup}}
  static void f();
};

struct X0 { };

template <class T> struct Derived: Base<int>, Base<char> {
  typename Derived::Base b;	// expected-error{{member 'Base' found in multiple base classes of different types}}
  typename Derived::Base<double> d;	// OK

  void g(X0 *t) {
    t->Derived::Base<T>::f();
    t->Base<T>::f();
    t->Base::f(); // expected-error{{member 'Base' found in multiple base classes of different types}}
  }
};

namespace PR6717 {
  template <typename T>
  class WebVector {
  } // expected-error {{expected ';' after class}}

    WebVector(const WebVector<T>& other) { } // expected-error{{undeclared identifier 'T'}} \
                                                precxx17-error{{a type specifier is required}} \
                                                cxx17-error{{deduction guide declaration without trailing return type}}

  template <typename C>
  WebVector<T>& operator=(const C& other) { } // expected-error{{undeclared identifier 'T'}}
}
