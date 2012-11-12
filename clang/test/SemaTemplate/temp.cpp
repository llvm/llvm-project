// RUN: %clang_cc1 -fsyntax-only -verify %s


namespace test0 {
  // p3
  template<typename T> int foo(T), bar(T, T); // expected-error{{single entity}}
}

// PR7252
namespace test1 {
  namespace A { template<typename T> struct Base { typedef T t; }; } // expected-note {{member found}}
  namespace B { template<typename T> struct Base { typedef T t; }; } // expected-note {{member found}}

  template<typename T> struct Derived : A::Base<char>, B::Base<int> {
    // FIXME: the syntax error here is unfortunate
    typename Derived::Base<float>::t x; // expected-error {{found in multiple base classes of different types}} \
                                        // expected-error {{expected member name or ';'}}
  };
}
