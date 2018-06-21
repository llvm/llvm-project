// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// The nested-name-specifier of a qualified declarator-id shall not begin with a decltype-specifier.
class foo {
  static int i;
  void func();
};

int decltype(foo())::i; // expected-error{{'decltype' cannot be used to name a declaration}}
void decltype(foo())::func() { // expected-error{{'decltype' cannot be used to name a declaration}}
}


template<typename T>
class tfoo {
  static int i;
  void func();
};

template<typename T>
int decltype(tfoo<T>())::i; // expected-error{{nested name specifier 'decltype(tfoo<T>())::' for declaration does not refer into a class, class template or class template partial specialization}}
template<typename T>
void decltype(tfoo<T>())::func() { // expected-error{{nested name specifier 'decltype(tfoo<T>())::' for declaration does not refer into a class, class template or class template partial specialization}}
}

// An init-declarator named with a qualified-id can refer to an element of the
// inline namespace set of the named namespace.
namespace inline_namespaces {
  namespace N {
    inline namespace M {
      void f(); // expected-note {{possible target}}
      void g();
      extern int m; // expected-note {{candidate}}
      extern int n;
      struct S; // expected-note {{candidate}}
      struct T;
      enum E : int; // expected-note {{candidate}}
      enum F : int;
      template<typename T> void ft(); // expected-note {{here}}
      template<typename T> void gt(); // expected-note {{here}}
      template<typename T> extern int mt; // expected-note {{here}} expected-warning {{extension}}
      template<typename T> extern int nt; // expected-note {{here}} expected-warning {{extension}}
      template<typename T> struct U; // expected-note {{here}}
      template<typename T> struct V; // expected-note {{here}}
    }

    // When named by unqualified-id, we do *not* look in the inline namespace
    // set.
    void f() {} // expected-note {{possible target}}
    int m; // expected-note {{candidate}}
    struct S {}; // expected-note {{candidate}}
    enum E : int {}; // expected-note {{candidate}}

    static_assert(&f != &M::f, ""); // expected-error {{reference to overloaded function could not be resolved}}
    static_assert(&m != &M::m, ""); // expected-error {{ambiguous}}
    typedef S X; // expected-error {{ambiguous}}
    typedef E Y; // expected-error {{ambiguous}}

    // When named by (unqualified) template-id, we do look in the inline
    // namespace set.  See [namespace.def]p8, [temp.explicit]p3,
    // [temp.expl.spec]p2.
    //
    // This is not explicitly specified for partial specializations, but
    // that is just a language defect.
    template<> void ft<int>() {}
    template void ft<char>(); // expected-error {{undefined}}

    template<typename T> int mt<T*>;
    template<> int mt<int>;
    template int mt<int*>;
    template int mt<char>; // expected-error {{undefined}}

    template<typename T> struct U<T*> {};
    template<> struct U<int> {};
    template struct U<int*>;
    template struct U<char>; // expected-error {{undefined}}
  }

  // When named by qualified-id, we *do* look in the inline namespace set.
  void N::g() {}
  int N::n;
  struct N::T {};
  enum N::F : int {};

  static_assert(&N::g == &N::M::g, "");
  static_assert(&N::n == &N::M::n, "");
  typedef N::T X;
  typedef N::M::T X;
  typedef N::F Y;
  typedef N::M::F Y;

  template<> void N::gt<int>() {}
  template void N::gt<char>(); // expected-error {{undefined}}

  template<typename T> int N::nt<T*>;
  template<> int N::nt<int>;
  template int N::nt<int*>;
  template int N::nt<char>; // expected-error {{undefined}}

  template<typename T> struct N::V<T*> {};
  template<> struct N::V<int> {};
  template struct N::V<int*>;
  template struct N::V<char>; // expected-error {{undefined}}
}
