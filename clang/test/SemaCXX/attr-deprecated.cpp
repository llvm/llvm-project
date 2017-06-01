// RUN: %clang_cc1 %s -verify -fexceptions
class A {
  void f() __attribute__((deprecated)); // expected-note 2 {{'f' has been explicitly marked deprecated here}}
  void g(A* a);
  void h(A* a) __attribute__((deprecated));

  int b __attribute__((deprecated)); // expected-note 2 {{'b' has been explicitly marked deprecated here}}
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
  virtual void f() __attribute__((deprecated)); // expected-note 6 {{'f' has been explicitly marked deprecated here}}
  void g();
};

void B::g() {
  f(); // expected-warning{{'f' is deprecated}}
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
  b->f(); // expected-warning{{'f' is deprecated}}
  b->B::f(); // expected-warning{{'f' is deprecated}}
  
  c->f();
  c->C::f();
  c->B::f(); // expected-warning{{'f' is deprecated}}
}

struct D {
  virtual void f() __attribute__((deprecated));// expected-note{{'f' has been explicitly marked deprecated here}}
  virtual void f(int) __attribute__((deprecated));// expected-note{{'f' has been explicitly marked deprecated here}}
  virtual void f(int, int) __attribute__((deprecated));// expected-note{{'f' has been explicitly marked deprecated here}}
};

void D::f() { } 
void D::f(int v) { } 
void D::f(int v1, int v2) { } 

void f(D* d) {
  d->f(); // expected-warning{{'f' is deprecated}}
  d->f(42); // expected-warning{{'f' is deprecated}}
  d->f(42, 24); // expected-warning{{'f' is deprecated}}
}


// Overloaded namespace members.
namespace test1 {
  void foo(int) __attribute__((deprecated)); // expected-note {{'foo' has been explicitly marked deprecated here}}
  void test1() { foo(10); } // expected-warning {{deprecated}}
  void foo(short) __attribute__((deprecated)); // expected-note {{'foo' has been explicitly marked deprecated here}}
  void test2(short s) { foo(s); } // expected-warning {{deprecated}}
  void foo(long);
  void test3(long l) { foo(l); }
  struct A {
    friend void foo(A*) __attribute__((deprecated)); // expected-note {{'foo' has been explicitly marked deprecated here}}
  };
  void test4(A *a) { foo(a); } // expected-warning {{deprecated}}

  namespace ns {
    struct Foo {};
    void foo(const Foo &f) __attribute__((deprecated)); // expected-note {{'foo' has been explicitly marked deprecated here}}
  }
  void test5() {
    foo(ns::Foo()); // expected-warning {{deprecated}}
  }
}

// Overloaded class members.
namespace test2 {
  struct A {
    void foo(int) __attribute__((deprecated)); // expected-note 2 {{'foo' has been explicitly marked deprecated here}}
    void foo(long);
    static void bar(int) __attribute__((deprecated)); // expected-note 3 {{'bar' has been explicitly marked deprecated here}}
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
    void operator*(int) __attribute__((deprecated)); // expected-note {{'operator*' has been explicitly marked deprecated here}}
    void operator-(const A &) const;
  };
  void operator+(const A &, const A &);
  void operator+(const A &, int) __attribute__((deprecated)); // expected-note {{'operator+' has been explicitly marked deprecated here}}
  void operator-(const A &, int) __attribute__((deprecated)); // expected-note {{'operator-' has been explicitly marked deprecated here}}

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
    operator intfn() __attribute__((deprecated)); // expected-note {{'operator void (*)(int)' has been explicitly marked deprecated here}}
    operator unintfn();
    void operator ()(A &) __attribute__((deprecated)); // expected-note {{'operator()' has been explicitly marked deprecated here}}
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
    operator int() __attribute__((deprecated)); // expected-note 3 {{'operator int' has been explicitly marked deprecated here}}
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
  enum __attribute__((deprecated)) A { // expected-note {{'A' has been explicitly marked deprecated here}}
    a0 // expected-note {{'a0' has been explicitly marked deprecated here}}
  };
  void testA() {
    A x; // expected-warning {{'A' is deprecated}}
    x = a0; // expected-warning {{'a0' is deprecated}}
  }
  
  enum B {
    b0 __attribute__((deprecated)), // expected-note {{'b0' has been explicitly marked deprecated here}}
    b1
  };
  void testB() {
    B x;
    x = b0; // expected-warning {{'b0' is deprecated}}
    x = b1;
  }

  template <class T> struct C {
    enum __attribute__((deprecated)) Enum { // expected-note {{'Enum' has been explicitly marked deprecated here}}
      c0 // expected-note {{'c0' has been explicitly marked deprecated here}}
    };
  };
  void testC() {
    C<int>::Enum x; // expected-warning {{'Enum' is deprecated}}
    x = C<int>::c0; // expected-warning {{'c0' is deprecated}}
  }

  template <class T> struct D {
    enum Enum {
      d0,
      d1 __attribute__((deprecated)), // expected-note {{'d1' has been explicitly marked deprecated here}}
    };
  };
  void testD() {
    D<int>::Enum x;
    x = D<int>::d0;
    x = D<int>::d1; // expected-warning {{'d1' is deprecated}}
  }
}

namespace test7 {
  struct X {
    void* operator new(typeof(sizeof(void*))) __attribute__((deprecated));  // expected-note{{'operator new' has been explicitly marked deprecated here}}
    void operator delete(void *) __attribute__((deprecated));  // expected-note{{'operator delete' has been explicitly marked deprecated here}}
  };

  void test() {
    X *x = new X;  // expected-warning{{'operator new' is deprecated}} expected-warning{{'operator delete' is deprecated}}
  }
}

// rdar://problem/15044218
typedef struct TDS {
} TDS __attribute__((deprecated)); // expected-note {{'TDS' has been explicitly marked deprecated here}}
TDS tds; // expected-warning {{'TDS' is deprecated}}
struct TDS tds2; // no warning, attribute only applies to the typedef.
