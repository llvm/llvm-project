// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Opaque0 {};
struct Opaque1 {};

// Redeclarations are okay in a namespace.
namespace test0 {
  namespace ns {
    void foo(Opaque0); // expected-note 2 {{candidate function}}
  }

  using ns::foo;
  using ns::foo;

  void test0() {
    foo(Opaque1()); // expected-error {{no matching function for call}}
  }

  namespace ns {
    void foo(Opaque1);
  }

  void test1() {
    foo(Opaque1()); // expected-error {{no matching function for call}}
  }

  using ns::foo;

  void test2() {
    foo(Opaque1());
  }

  using ns::foo;
}

// Make sure we handle transparent contexts the same way.
namespace test1 {
  namespace ns {
    void foo(Opaque0); // expected-note 2 {{candidate function}}
  }

  extern "C++" {
    using ns::foo;
  }

  void test0() {
    foo(Opaque1()); // expected-error {{no matching function for call}}
  }

  namespace ns {
    void foo(Opaque1);
  }

  void test1() {
    foo(Opaque1()); // expected-error {{no matching function for call}}
  }

  extern "C++" {
    using ns::foo;
  }

  void test2() {
    foo(Opaque1());
  }
}

// Make sure we detect invalid redeclarations that can't be detected
// until template instantiation.
namespace test2 {
  template <class T> struct Base {
    typedef Base type;
    void foo();
  };

  template <class T> struct Derived : Base<T> {
    // These are invalid redeclarations, detectable only after
    // instantiation.
    using Base<T>::foo; // expected-note {{previous using decl}}
    using Base<T>::type::foo; //expected-error {{redeclaration of using decl}}
  };

  template struct Derived<int>; // expected-note {{in instantiation of template class}}
}

// Redeclarations are okay in a function.
namespace test3 {
  namespace N {
    int f(int);
    typedef int type;
  }

  void g() {
    using N::f;
    using N::f;
    using N::type;
    using N::type;
  }
}
