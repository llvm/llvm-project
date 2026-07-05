// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// We have to avoid ADL for this test.

template <unsigned N> class test {};

class foo {};	// expected-note {{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
#endif
test<0> foo(foo); // expected-note {{candidate}}

namespace Test0 {
  class foo { int x; };
  test<1> foo(class foo);

  namespace A {
    test<2> foo(class ::foo); // expected-note {{candidate}} \
    // expected-note{{passing argument to parameter here}}

    void test0() {
      using ::foo;

      class foo a;
      test<0> _ = (foo)(a);
    }

    void test1() {
      using Test0::foo;

      class foo a;
      test<1> _ = (foo)(a);
    };

    void test2() {
      class ::foo a;
      
      // Argument-dependent lookup is ambiguous between B:: and ::.
      test<0> _0 = foo(a); // expected-error {{call to 'foo' is ambiguous}}

      // But basic unqualified lookup is not.
      test<2> _1 = (foo)(a);

      class Test0::foo b;
      test<2> _2 = (foo)(b); // expected-error {{no viable conversion from 'class Test0::foo' to 'class ::foo'}}
    }
  }
}

namespace Test1 {
  namespace A {
    class a {};
  }

  namespace B {
    typedef class {} b;
  }

  namespace C {
    int c(); // expected-note {{target of using declaration}}
  }

  namespace D {
    using typename A::a;
    using typename B::b;
    using typename C::c; // expected-error {{'typename' keyword used on a non-type}}

    a _1 = A::a();
    b _2 = B::b();
  }
}

namespace test2 {
  class A {
  protected:
    operator int();
    operator bool();
  };

  class B : private A {
  protected:
    using A::operator int; // expected-note {{declared protected here}}
  public:
    using A::operator bool;
  };

  int test() {
    bool b = B();
    return B(); // expected-error {{'operator int' is a protected member of 'test2::B'}}
  }
}

namespace test3 {
  class A {
  public:
    ~A();
  };

  class B {
    friend class C;
  private:
    operator A*();
  };

  class C : public B {
  public:
    using B::operator A*;
  };

  void test() {
    delete C();
  }
}
