// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++03 [namespace.udecl]p12:
//   When a using-declaration brings names from a base class into a
//   derived class scope, member functions in the derived class
//   override and/or hide member functions with the same name and
//   parameter types in a base class (rather than conflicting).

template <unsigned n> struct Opaque {};
template <unsigned n> void expect(Opaque<n> _) {}

// PR5727
// This just shouldn't crash.
namespace test0 {
  template<typename> struct RefPtr { };
  template<typename> struct PtrHash {
    static void f() { }
  };
  template<typename T> struct PtrHash<RefPtr<T> > : PtrHash<T*> {
    using PtrHash<T*>::f;
    static void f() { f(); }
  };
}

// Simple hiding.
namespace test1 {
  struct Base {
    Opaque<0> foo(Opaque<0>);
    Opaque<0> foo(Opaque<1>);
    Opaque<0> foo(Opaque<2>);
  };

  // using before decls
  struct Test0 : Base {
    using Base::foo;
    Opaque<1> foo(Opaque<1>);
    Opaque<1> foo(Opaque<3>);

    void test0() { Opaque<0> _ = foo(Opaque<0>()); }
    void test1() { Opaque<1> _ = foo(Opaque<1>()); }
    void test2() { Opaque<0> _ = foo(Opaque<2>()); }
    void test3() { Opaque<1> _ = foo(Opaque<3>()); }
  };

  // using after decls
  struct Test1 : Base {
    Opaque<1> foo(Opaque<1>);
    Opaque<1> foo(Opaque<3>);
    using Base::foo;

    void test0() { Opaque<0> _ = foo(Opaque<0>()); }
    void test1() { Opaque<1> _ = foo(Opaque<1>()); }
    void test2() { Opaque<0> _ = foo(Opaque<2>()); }
    void test3() { Opaque<1> _ = foo(Opaque<3>()); }
  };

  // using between decls
  struct Test2 : Base {
    Opaque<1> foo(Opaque<0>);
    using Base::foo;
    Opaque<1> foo(Opaque<2>);
    Opaque<1> foo(Opaque<3>);

    void test0() { Opaque<1> _ = foo(Opaque<0>()); }
    void test1() { Opaque<0> _ = foo(Opaque<1>()); }
    void test2() { Opaque<1> _ = foo(Opaque<2>()); }
    void test3() { Opaque<1> _ = foo(Opaque<3>()); }
  };
}

// Crazy dependent hiding.
namespace test2 {
  struct Base {
    void foo(int);
  };

  template <typename T> struct Derived1 : Base {
    using Base::foo;
    void foo(T);

    void testUnresolved(int i) { foo(i); }
  };

  void test0(int i) {
    Derived1<int> d1;
    d1.foo(i);
    d1.testUnresolved(i);
  }

  // Same thing, except with the order of members reversed.
  template <typename T> struct Derived2 : Base {
    void foo(T);
    using Base::foo;

    void testUnresolved(int i) { foo(i); }
  };

  void test1(int i) {
    Derived2<int> d2;
    d2.foo(i);
    d2.testUnresolved(i);
  }
}

// Hiding of member templates.
namespace test3 {
  struct Base {
    template <class T> Opaque<0> foo() { return Opaque<0>(); }
    template <int n> Opaque<1> foo() { return Opaque<1>(); }
  };

  struct Derived1 : Base {
    using Base::foo;
    template <int n> Opaque<2> foo() { return Opaque<2>(); } // expected-note {{invalid explicitly-specified argument for template parameter 'n'}}
  };

  struct Derived2 : Base {
    template <int n> Opaque<2> foo() { return Opaque<2>(); } // expected-note {{invalid explicitly-specified argument for template parameter 'n'}}
    using Base::foo;
  };

  struct Derived3 : Base {
    using Base::foo;
    template <class T> Opaque<3> foo() { return Opaque<3>(); } // expected-note {{invalid explicitly-specified argument for template parameter 'T'}}
  };

  struct Derived4 : Base {
    template <class T> Opaque<3> foo() { return Opaque<3>(); } // expected-note {{invalid explicitly-specified argument for template parameter 'T'}}
    using Base::foo;
  };

  void test() {
    expect<0>(Base().foo<int>());
    expect<1>(Base().foo<0>());
    expect<0>(Derived1().foo<int>()); // expected-error {{no matching member function for call to 'foo'}}
    expect<2>(Derived1().foo<0>());
    expect<0>(Derived2().foo<int>()); // expected-error {{no matching member function for call to 'foo'}}
    expect<2>(Derived2().foo<0>());
    expect<3>(Derived3().foo<int>());
    expect<1>(Derived3().foo<0>()); // expected-error {{no matching member function for call to 'foo'}}
    expect<3>(Derived4().foo<int>());
    expect<1>(Derived4().foo<0>()); // expected-error {{no matching member function for call to 'foo'}}
  }
}

// PR7384: access control for member templates.
namespace test4 {
  class Base {
  protected:
    template<typename T> void foo(T);
    template<typename T> void bar(T); // expected-note {{declared protected here}}
  };

  struct Derived : Base {
    using Base::foo;
  };

  void test() {
    Derived d;
    d.foo<int>(3);
    d.bar<int>(3); // expected-error {{'bar' is a protected member}}
  }
}

namespace test5 {
  struct Derived;
  struct Base {
    void operator=(const Derived&);
  };
  struct Derived : Base {
    // Hidden by implicit derived class operator.
    using Base::operator=;
  };
  void f(Derived d) {
    d = d;
  }
}

#if __cplusplus >= 201103L
namespace test6 {
  struct Derived;
  struct Base {
    void operator=(Derived&&);
  };
  struct Derived : Base {
    // Hidden by implicit derived class operator.
    using Base::operator=;
  };
  void f(Derived d) {
    d = Derived();
  }
}
#endif
