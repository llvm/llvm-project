// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  class A {
    protected: int x; // expected-note 3 {{declared}} \
    // expected-note {{member is declared here}}
    static int sx; // expected-note 3 {{declared}} \
    // expected-note {{member is declared here}}
  };
  class B : public A {
  };
  class C : protected A { // expected-note {{declared}}
  };
  class D : private B { // expected-note 3 {{constrained}}
  };

  void test(A &a) {
    (void) a.x; // expected-error {{'x' is a protected member}}
    (void) a.sx; // expected-error {{'sx' is a protected member}}
  }
  void test(B &b) {
    (void) b.x; // expected-error {{'x' is a protected member}}
    (void) b.sx; // expected-error {{'sx' is a protected member}}
  }
  void test(C &c) {
    (void) c.x; // expected-error {{'x' is a protected member}} expected-error {{protected base class}}
    (void) c.sx; // expected-error {{'sx' is a protected member}}
  }
  void test(D &d) {
    (void) d.x; // expected-error {{'x' is a private member}} expected-error {{private base class}}
    (void) d.sx; // expected-error {{'sx' is a private member}}
  }
}

namespace test1 {
  class A {
    protected: int x;
    static int sx;
    static void test(A&);
  };
  class B : public A {
    static void test(B&);
  };
  class C : protected A {
    static void test(C&);
  };
  class D : private B {
    static void test(D&);
  };

  void A::test(A &a) {
    (void) a.x;
    (void) a.sx;
  }
  void B::test(B &b) {
    (void) b.x;
    (void) b.sx;
  }
  void C::test(C &c) {
    (void) c.x;
    (void) c.sx;
  }
  void D::test(D &d) {
    (void) d.x;
    (void) d.sx;
  }
}

namespace test2 {
  class A {
    protected: int x; // expected-note 3 {{can only access this member on an object of type}}
    static int sx;
    static void test(A&);
  };
  class B : public A {
    static void test(A&);
  };
  class C : protected A {
    static void test(A&);
  };
  class D : private B {
    static void test(A&);
  };

  void A::test(A &a) {
    (void) a.x;
    (void) a.sx;
  }
  void B::test(A &a) {
    (void) a.x; // expected-error {{'x' is a protected member}}
    (void) a.sx;
  }
  void C::test(A &a) {
    (void) a.x; // expected-error {{'x' is a protected member}}
    (void) a.sx;
  }
  void D::test(A &a) {
    (void) a.x; // expected-error {{'x' is a protected member}}
    (void) a.sx;
  }
}

namespace test3 {
  class B;
  class A {
    protected: int x; //expected-note {{declared protected}} // expected-note {{can only access this member on an object of type}}
    static int sx;
    static void test(B&);
  };
  class B : public A {
    static void test(B&);
  };
  class C : protected A {
    static void test(B&);
  };
  class D : private B {
    static void test(B&);
  };

  void A::test(B &b) {
    (void) b.x;
    (void) b.sx;
  }
  void B::test(B &b) {
    (void) b.x;
    (void) b.sx;
  }
  void C::test(B &b) {
    (void) b.x; // expected-error {{'x' is a protected member}}
    (void) b.sx;
  }
  void D::test(B &b) {
    (void) b.x; // expected-error {{'x' is a protected member}}
    (void) b.sx;
  }
}

namespace test4 {
  class C;
  class A {
    protected: int x; // expected-note 2{{declared protected here}} expected-note{{member is declared here}}
    static int sx;    // expected-note 3{{member is declared here}}
    static void test(C&);
  };
  class B : public A {
    static void test(C&);
  };
  class C : protected A { // expected-note 4 {{constrained}} expected-note 3 {{declared}}
    static void test(C&);
  };
  class D : private B {
    static void test(C&);
  };

  void A::test(C &c) {
    (void) c.x;  // expected-error {{'x' is a protected member}} \
                 // expected-error {{protected base class}}
    (void) c.sx; // expected-error {{'sx' is a protected member}}
  }
  void B::test(C &c) {
    (void) c.x;  // expected-error {{'x' is a protected member}} \
                 // expected-error {{protected base class}}
    (void) c.sx; // expected-error {{'sx' is a protected member}}
  }
  void C::test(C &c) {
    (void) c.x;
    (void) c.sx;
  }
  void D::test(C &c) {
    (void) c.x;  // expected-error {{'x' is a protected member}} \
                 // expected-error {{protected base class}}
    (void) c.sx; // expected-error {{'sx' is a protected member}}
  }
}

namespace test5 {
  class D;
  class A {
    protected: int x; // expected-note 3{{member is declared here}}
    static int sx; // expected-note 3{{member is declared here}}
    static void test(D&);
  };
  class B : public A {
    static void test(D&);
  };
  class C : protected A {
    static void test(D&);
  };
  class D : private B { // expected-note 9 {{constrained}}
    static void test(D&);
  };

  void A::test(D &d) {
    (void) d.x;  // expected-error {{'x' is a private member}} \
                 // expected-error {{cannot cast}}
    (void) d.sx; // expected-error {{'sx' is a private member}}
  }
  void B::test(D &d) {
    (void) d.x;  // expected-error {{'x' is a private member}} \
                 // expected-error {{cannot cast}}
    (void) d.sx; // expected-error {{'sx' is a private member}}
  }
  void C::test(D &d) {
    (void) d.x;  // expected-error {{'x' is a private member}} \
                 // expected-error {{cannot cast}}
    (void) d.sx; // expected-error {{'sx' is a private member}}
  }
  void D::test(D &d) {
    (void) d.x;
    (void) d.sx;
  }
}

namespace test6 {
  class Static {};
  class A {
  protected:
    void foo(int); // expected-note 3 {{can only access this member on an object of type}}
    void foo(long);
    static void foo(Static);

    static void test(A&);
  };
  class B : public A {
    static void test(A&);
  };
  class C : protected A {
    static void test(A&);
  };
  class D : private B {
    static void test(A&);
  };

  void A::test(A &a) {
    a.foo(10);
    a.foo(Static());
  }
  void B::test(A &a) {
    a.foo(10); // expected-error {{'foo' is a protected member}}
    a.foo(Static());
  }
  void C::test(A &a) {
    a.foo(10); // expected-error {{'foo' is a protected member}}
    a.foo(Static());
  }
  void D::test(A &a) {
    a.foo(10); // expected-error {{'foo' is a protected member}}
    a.foo(Static());
  }
}

namespace test7 {
  class Static {};
  class A {
    protected:
    void foo(int); // expected-note 3 {{must name member using the type of the current context}}
    void foo(long);
    static void foo(Static);

    static void test();
  };
  class B : public A {
    static void test();
  };
  class C : protected A {
    static void test();
  };
  class D : private B {
    static void test();
  };

  void A::test() {
    void (A::*x)(int) = &A::foo;
    void (*sx)(Static) = &A::foo;
  }
  void B::test() {
    void (A::*x)(int) = &A::foo; // expected-error {{'foo' is a protected member}}
    void (*sx)(Static) = &A::foo;
  }
  void C::test() {
    void (A::*x)(int) = &A::foo; // expected-error {{'foo' is a protected member}}
    void (*sx)(Static) = &A::foo;
  }
  void D::test() {
    void (A::*x)(int) = &A::foo; // expected-error {{'foo' is a protected member}}
    void (*sx)(Static) = &A::foo;
  }
}

namespace test8 {
  class Static {};
  class A {
    protected:
    void foo(int); // expected-note 3 {{must name member using the type of the current context}}
    void foo(long);
    static void foo(Static);

    static void test();
  };
  class B : public A {
    static void test();
  };
  class C : protected A {
    static void test();
  };
  class D : private B {
    static void test();
  };
  void call(void (A::*)(int));
  void calls(void (*)(Static));

  void A::test() {
    call(&A::foo);
    calls(&A::foo);
  }
  void B::test() {
    call(&A::foo); // expected-error {{'foo' is a protected member}}
    calls(&A::foo);
  }
  void C::test() {
    call(&A::foo); // expected-error {{'foo' is a protected member}}
    calls(&A::foo);
  }
  void D::test() {
    call(&A::foo); // expected-error {{'foo' is a protected member}}
    calls(&A::foo);
  }
}

namespace test9 {
  class A { // expected-note {{member is declared here}}
  protected: int foo(); // expected-note 4 {{declared}} expected-note 2 {{can only access this member on an object of type}} expected-note {{member is declared here}}
  };

  class B : public A { // expected-note {{member is declared here}}
    friend class D;
  };

  class C : protected B { // expected-note {{declared}} \
                          // expected-note 9 {{constrained}}
  };

  class D : public A {
    static void test(A &a) {
      a.foo(); // expected-error {{'foo' is a protected member}}
      a.A::foo(); // expected-error {{'foo' is a protected member}}
      a.B::foo();
      a.C::foo(); // expected-error {{'foo' is a protected member}}
    }

    static void test(B &b) {
      b.foo();
      b.A::foo();
      b.B::foo();
      b.C::foo(); // expected-error {{'foo' is a protected member}}
    }

    static void test(C &c) {
      c.foo();    // expected-error {{'foo' is a protected member}} \
                  // expected-error {{cannot cast}}
      c.A::foo(); // expected-error {{'A' is a protected member}} \
                  // expected-error {{cannot cast}}
      c.B::foo(); // expected-error {{'B' is a protected member}} \
                  // expected-error {{cannot cast}}
      c.C::foo(); // expected-error {{'foo' is a protected member}} \
                  // expected-error {{cannot cast}}
    }

    static void test(D &d) {
      d.foo();
      d.A::foo();
      d.B::foo();
      d.C::foo(); // expected-error {{'foo' is a protected member}}
    }
  };
}

namespace test10 {
  template<typename T> class A {
  protected:
    int foo();
    int foo() const;

    ~A() { foo(); }
  };

  template class A<int>;
}

// rdar://problem/8360285: class.protected friendship
namespace test11 {
  class A {
  protected:
    int foo();
  };

  class B : public A {
    friend class C;
  };

  class C {
    void test() {
      B b;
      b.A::foo();
    }
  };
}

// This friendship is considered because a public member of A would be
// a private member of C.
namespace test12 {
  class A { protected: int foo(); };
  class B : public virtual A {};
  class C : private B { friend void test(); };
  class D : private C, public virtual A {};

  void test() {
    D d;
    d.A::foo();
  }
}

// This friendship is not considered because a public member of A is
// inaccessible in C.
namespace test13 {
  class A { protected: int foo(); }; // expected-note {{declared protected here}}
  class B : private virtual A {};
  class C : private B { friend void test(); };
  class D : public virtual A {};

  void test() {
    D d;
    d.A::foo(); // expected-error {{protected member}}
  }
}

// PR8058
namespace test14 {
  class A {
  protected:
    template <class T> void temp(T t); // expected-note {{must name member using the type of the current context}}

    void nontemp(int); // expected-note {{must name member using the type of the current context}}

    template <class T> void ovl_temp(T t); // expected-note {{must name member using the type of the current context}}
    void ovl_temp(float);

    void ovl_nontemp(int); // expected-note {{must name member using the type of the current context}}
    void ovl_nontemp(float);

    template <class T> void ovl_withtemp(T);
    void ovl_withtemp(int); // expected-note {{must name member using the type of the current context}}
  };

  class B : public A {
    void use() {
      void (A::*ptr)(int);
      ptr = &A::temp; // expected-error {{protected member}}
      ptr = &A::nontemp; // expected-error {{protected member}}
      ptr = &A::ovl_temp; // expected-error {{protected member}}
      ptr = &A::ovl_nontemp; // expected-error {{protected member}}
      ptr = &A::ovl_withtemp; // expected-error {{protected member}}
    }
  };
}

namespace test15 {
  class A {
  protected:
    A(); // expected-note 2 {{protected constructor can only be used to construct a base class subobject}}
    A(const A &); // expected-note {{protected constructor can only be used to construct a base class subobject}}
    ~A(); // expected-note 3 {{protected destructor can only be used to destroy a base class subobject}}
  };

  class B : public A {
    // The uses here are fine.
    B() {}
    B(int i) : A() {}
    ~B() {}

    // All these uses are bad.

    void test0() {
      A a; // expected-error {{protected constructor}} expected-error {{protected destructor}}
    }

    A *test1() {
      return new A(); // expected-error {{protected constructor}}
    }

    void test2(A *a) {
      delete a; // expected-error {{protected destructor}}
    }

    A test3(A *a) {
      return *a; // expected-error {{protected constructor}}
    }

    void test4(A *a) {
      a->~A(); // expected-error {{protected member}}
    }
  };
}

namespace test16 {
  class A {
  protected:
    ~A();
  };

  class B : public virtual A {
  public:
    ~B() {}
  };

  class C : public B {
    ~C() {}
  };
}
