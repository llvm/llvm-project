// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T> class vector2 {};
template<typename T> class vector : vector2<T> {};

template<typename T> void Foo2(vector2<const T*> V) {}  // expected-note{{candidate template ignored: cannot deduce a type for 'T' that would make 'const T' equal 'int'}}
template<typename T> void Foo(vector<const T*> V) {} // expected-note {{candidate template ignored: cannot deduce a type for 'T' that would make 'const T' equal 'int'}}

void test() {
  Foo2(vector2<int*>());  // expected-error{{no matching function for call to 'Foo2'}}
  Foo(vector<int*>());  // expected-error{{no matching function for call to 'Foo'}}
}

namespace rdar13267210 {
  template < typename T > class A {
    BaseTy; // expected-error{{C++ requires a type specifier for all declarations}}
  };

  template < typename T, int N > class C: A < T > {};

  class B {
    C<long, 16> ExternalDefinitions;
    C<long, 64> &Record;

    void AddSourceLocation(A<long> &R); // expected-note{{passing argument to parameter 'R' here}}
    void AddTemplateKWAndArgsInfo() {
      AddSourceLocation(Record); // expected-error{{non-const lvalue reference to type}}
    }
  };
}

namespace PR16292 {
  class IncompleteClass;  // expected-note{{forward declaration}}
  class BaseClass {
    IncompleteClass Foo;  // expected-error{{field has incomplete type}}
  };
  template<class T> class DerivedClass : public BaseClass {};
  void* p = new DerivedClass<void>;
}

namespace rdar14183893 {
  class Typ { // expected-note {{not complete}}
    Typ x; // expected-error {{incomplete type}}
  };

  template <unsigned  C> class B :  Typ {};
  typedef B<0> TFP;

  class A {
    TFP m_p;
    void Enable() { 0, A(); } // expected-warning {{unused}}
  };
}
