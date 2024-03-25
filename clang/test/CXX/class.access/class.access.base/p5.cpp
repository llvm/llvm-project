// RUN: %clang_cc1 -verify %s

namespace test0 {
  struct A {
    static int x;
  };
  struct B : A {};
  struct C : B {};

  int test() {
    return A::x
         + B::x
         + C::x;
  }
}

namespace test1 {
  struct A {
    private: static int x; // expected-note 5 {{declared private here}}
    static int test() { return x; }
  };
  struct B : public A {
    static int test() { return x; } // expected-error {{private member}}
  };
  struct C : private A {
    static int test() { return x; } // expected-error {{private member}}
  };

  struct D {
    public: static int x; // expected-note{{member is declared here}}
    static int test() { return x; }
  };
  struct E : private D { // expected-note{{constrained by private inheritance}}
    static int test() { return x; }
  };

  int test() {
    return A::x // expected-error {{private member}}
         + B::x // expected-error {{private member}}
         + C::x // expected-error {{private member}}
         + D::x
         + E::x; // expected-error {{private member}}
  }
}

namespace test2 {
  class A {
  protected: static int x; // expected-note{{member is declared here}}
  };

  class B : private A {}; // expected-note {{private inheritance}}
  class C : private A {
    int test(B *b) {
      return b->x; // expected-error {{private member}}
    }
  };
}

namespace test3 {
  class A {
  protected: static int x;
  };

  class B : public A {};
  class C : private A {
    int test(B *b) {
      // x is accessible at C when named in A.
      // A is an accessible base of B at C.
      // Therefore this succeeds.
      return b->x;
    }
  };
}

// Don't crash.
// Note that 'field' is indeed a private member of X but that access
// is indeed ultimately constrained by the protected inheritance from Y.
// If someone wants to put the effort into improving this diagnostic,
// they can feel free; even explaining it in person would be a pain.
namespace test4 {
  class Z;
  class X {
  public:
    void f(Z *p);

  private:
    int field; // expected-note {{member is declared here}}
  };

  class Y : public X { };
  class Z : protected Y { }; // expected-note {{constrained by protected inheritance here}}

  void X::f(Z *p) {
    p->field = 0; // expected-error {{'field' is a private member of 'test4::X'}}
  }
}

// TODO: flesh out these cases
