// RUN: %clang_cc1 %s -verify -fsyntax-only
class A {
  void f() __attribute__((deprecated)); // expected-note 2 {{declared here}}
  void g(A* a);
  void h(A* a) __attribute__((deprecated));

  int b __attribute__((deprecated)); // expected-note 2 {{declared here}}
};

void A::g(A* a)
{
  f(); // expected-warning{{'f' is deprecated}}
  a->f(); // expected-warning{{'f' is deprecated}}
  
  (void)b; // expected-warning{{'b' is deprecated}}
  (void)a->b; // expected-warning{{'b' is deprecated}}
}

void A::h(A* a)
{
  f();
  a->f();
  
  (void)b;
  (void)a->b;
}

struct B {
  virtual void f() __attribute__((deprecated)); // expected-note 4 {{declared here}}
  void g();
};

void B::g() {
  f();
  B::f(); // expected-warning{{'f' is deprecated}}
}

struct C : B {
  virtual void f();
  void g();
};

void C::g() {
  f();
  C::f();
  B::f(); // expected-warning{{'f' is deprecated}}
}

void f(B* b, C *c) {
  b->f();
  b->B::f(); // expected-warning{{'f' is deprecated}}
  
  c->f();
  c->C::f();
  c->B::f(); // expected-warning{{'f' is deprecated}}
}

struct D {
  virtual void f() __attribute__((deprecated));
};

void D::f() { }

void f(D* d) {
  d->f();
}


// Overloaded namespace members.
namespace test1 {
  void foo(int) __attribute__((deprecated)); // expected-note {{declared here}}
  void test1() { foo(10); } // expected-warning {{deprecated}}
  void foo(short) __attribute__((deprecated)); // expected-note {{declared here}}
  void test2(short s) { foo(s); } // expected-warning {{deprecated}}
  void foo(long);
  void test3(long l) { foo(l); }
  struct A {
    friend void foo(A*) __attribute__((deprecated)); // expected-note {{declared here}}
  };
  void test4(A *a) { foo(a); } // expected-warning {{deprecated}}

  namespace ns {
    struct Foo {};
    void foo(const Foo &f) __attribute__((deprecated)); // expected-note {{declared here}}
  }
  void test5() {
    foo(ns::Foo()); // expected-warning {{deprecated}}
  }
}

// Overloaded class members.
namespace test2 {
  struct A {
    void foo(int) __attribute__((deprecated)); // expected-note 2 {{declared here}}
    void foo(long);
    static void bar(int) __attribute__((deprecated)); // expected-note 3 {{declared here}}
    static void bar(long);

    void test2(int i, long l);
  };
  void test1(int i, long l) {
    A a;
    a.foo(i); // expected-warning {{deprecated}}
    a.foo(l);
    a.bar(i); // expected-warning {{deprecated}}
    a.bar(l);
    A::bar(i); // expected-warning {{deprecated}}
    A::bar(l);
  }

  void A::test2(int i, long l) {
    foo(i); // expected-warning {{deprecated}}
    foo(l);
    bar(i); // expected-warning {{deprecated}}
    bar(l);
  }
}

// Overloaded operators.
namespace test3 {
  struct A {
    void operator*(const A &);
    void operator*(int) __attribute__((deprecated)); // expected-note {{declared here}}
    void operator-(const A &) const;
  };
  void operator+(const A &, const A &);
  void operator+(const A &, int) __attribute__((deprecated)); // expected-note {{declared here}}
  void operator-(const A &, int) __attribute__((deprecated)); // expected-note {{declared here}}

  void test() {
    A a, b;
    a + b;
    a + 1; // expected-warning {{deprecated}}
    a - b;
    a - 1; // expected-warning {{deprecated}}
    a * b;
    a * 1; // expected-warning {{deprecated}}
  }
}

// Overloaded operator call.
namespace test4 {
  struct A {
    typedef void (*intfn)(int);
    typedef void (*unintfn)(unsigned);
    operator intfn() __attribute__((deprecated)); // expected-note {{declared here}}
    operator unintfn();
    void operator ()(A &) __attribute__((deprecated)); // expected-note {{declared here}}
    void operator ()(const A &);
  };

  void test() {
    A a;
    a(1); // expected-warning {{deprecated}}
    a(1U);

    A &b = a;
    const A &c = a;
    a(b); // expected-warning {{deprecated}}
    a(c);
  }
}

namespace test5 {
  struct A {
    operator int() __attribute__((deprecated)); // expected-note 3 {{declared here}}
    operator long();
  };
  void test1(A a) {
    int i = a; // expected-warning {{deprecated}}
    long l = a;
  }

  void foo(int);
  void foo(void*);
  void bar(long);
  void bar(void*);
  void test2(A a) {
    foo(a); // expected-warning {{deprecated}}
    bar(a);
  }

  struct B {
    int myInt;
    long myLong;

    B(A &a) :
      myInt(a), // expected-warning {{deprecated}}
      myLong(a)
    {}
  };
}

// rdar://problem/8518751
namespace test6 {
  enum __attribute__((deprecated)) A { // expected-note {{declared here}}
    a0 // expected-note {{declared here}}
  };
  void testA() {
    A x; // expected-warning {{'A' is deprecated}}
    x = a0; // expected-warning {{'a0' is deprecated}}
  }
  
  enum B {
    b0 __attribute__((deprecated)), // expected-note {{declared here}}
    b1
  };
  void testB() {
    B x;
    x = b0; // expected-warning {{'b0' is deprecated}}
    x = b1;
  }

  template <class T> struct C {
    enum __attribute__((deprecated)) Enum { // expected-note {{declared here}}
      c0 // expected-note {{declared here}}
    };
  };
  void testC() {
    C<int>::Enum x; // expected-warning {{'Enum' is deprecated}}
    x = C<int>::c0; // expected-warning {{'c0' is deprecated}}
  }

  template <class T> struct D {
    enum Enum {
      d0,
      d1 __attribute__((deprecated)), // expected-note {{declared here}}
    };
  };
  void testD() {
    D<int>::Enum x;
    x = D<int>::d0;
    x = D<int>::d1; // expected-warning {{'d1' is deprecated}}
  }
}
