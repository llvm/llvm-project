// RUN: %clang_cc1 -fsyntax-only -Wall -Wuninitialized -Wno-unused-value -std=c++11 -verify %s

int foo(int x);
int bar(int* x);
int boo(int& x);
int far(const int& x);

// Test self-references within initializers which are guaranteed to be
// uninitialized.
int a = a; // no-warning: used to signal intended lack of initialization.
int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
int c = (c + c); // expected-warning 2 {{variable 'c' is uninitialized when used within its own initialization}}
int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

// Thes don't warn as they don't require the value.
int g = sizeof(g);
void* ptr = &ptr;
int h = bar(&h);
int i = boo(i);
int j = far(j);
int k = __alignof__(k);

int l = k ? l : l;  // expected-warning 2{{variable 'l' is uninitialized when used within its own initialization}}
int m = 1 + (k ? m : m);  // expected-warning 2{{variable 'm' is uninitialized when used within its own initialization}}
int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}

void test_stuff () {
  int a = a; // no-warning: used to signal intended lack of initialization.
  int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
  int c = (c + c); // expected-warning {{variable 'c' is uninitialized when used within its own initialization}}
  int d = ({ d + d ;}); // expected-warning {{variable 'd' is uninitialized when used within its own initialization}}
  int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
  int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

  // Thes don't warn as they don't require the value.
  int g = sizeof(g);
  void* ptr = &ptr;
  int h = bar(&h);
  int i = boo(i);
  int j = far(j);
  int k = __alignof__(k);

  int l = k ? l : l;  // FIXME: warn here
  int m = 1 + (k ? m : m);  // FIXME: warn here
  int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}

  for (;;) {
    int a = a; // no-warning: used to signal intended lack of initialization.
    int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
    int c = (c + c); // expected-warning {{variable 'c' is uninitialized when used within its own initialization}}
    int d = ({ d + d ;}); // expected-warning {{variable 'd' is uninitialized when used within its own initialization}}
    int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
    int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

    // Thes don't warn as they don't require the value.
    int g = sizeof(g);
    void* ptr = &ptr;
    int h = bar(&h);
    int i = boo(i);
    int j = far(j);
    int k = __alignof__(k);

    int l = k ? l : l;  // FIXME: warn here
    int m = 1 + (k ? m : m);  // FIXME: warn here
    int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
  }
}

// Test self-references with record types.
class A {
  // Non-POD class.
  public:
    enum count { ONE, TWO, THREE };
    int num;
    static int count;
    int get() const { return num; }
    int get2() { return num; }
    void set(int x) { num = x; }
    static int zero() { return 0; }

    A() {}
    A(A const &a) {}
    A(int x) {}
    A(int *x) {}
    A(A *a) {}
    ~A();
};

A getA() { return A(); }
A getA(int x) { return A(); }
A getA(A* a) { return A(); }
A getA(A a) { return A(); }

void setupA(bool x) {
  A a1;
  a1.set(a1.get());
  A a2(a1.get());
  A a3(a1);
  A a4(&a4);
  A a5(a5.zero());
  A a6(a6.ONE);
  A a7 = getA();
  A a8 = getA(a8.TWO);
  A a9 = getA(&a9);
  A a10(a10.count);

  A a11(a11);  // expected-warning {{variable 'a11' is uninitialized when used within its own initialization}}
  A a12(a12.get());  // expected-warning {{variable 'a12' is uninitialized when used within its own initialization}}
  A a13(a13.num);  // expected-warning {{variable 'a13' is uninitialized when used within its own initialization}}
  A a14 = A(a14);  // expected-warning {{variable 'a14' is uninitialized when used within its own initialization}}
  A a15 = getA(a15.num);  // expected-warning {{variable 'a15' is uninitialized when used within its own initialization}}
  A a16(&a16.num);  // expected-warning {{variable 'a16' is uninitialized when used within its own initialization}}
  A a17(a17.get2());  // expected-warning {{variable 'a17' is uninitialized when used within its own initialization}}
  A a18 = x ? a18 : a17;  // expected-warning {{variable 'a18' is uninitialized when used within its own initialization}}
  A a19 = getA(x ? a19 : a17);  // expected-warning {{variable 'a19' is uninitialized when used within its own initialization}}
  A a20{a20};  // expected-warning {{variable 'a20' is uninitialized when used within its own initialization}}
  A a21 = {a21};  // expected-warning {{variable 'a21' is uninitialized when used within its own initialization}}

  // FIXME: Make the local uninitialized warning consistant with the global
  // uninitialized checking.
  A *a22 = new A(a22->count);  // expected-warning {{variable 'a22' is uninitialized when used within its own initialization}}
  A *a23 = new A(a23->ONE);  // expected-warning {{variable 'a23' is uninitialized when used within its own initialization}}
  A *a24 = new A(a24->TWO);  // expected-warning {{variable 'a24' is uninitialized when used within its own initialization}}
  A *a25 = new A(a25->zero());  // expected-warning {{variable 'a25' is uninitialized when used within its own initialization}}

  A *a26 = new A(a26->get());    // expected-warning {{variable 'a26' is uninitialized when used within its own initialization}}
  A *a27 = new A(a27->get2());  // expected-warning {{variable 'a27' is uninitialized when used within its own initialization}}
  A *a28 = new A(a28->num);  // expected-warning {{variable 'a28' is uninitialized when used within its own initialization}}
}

bool x;

A a1;
A a2(a1.get());
A a3(a1);
A a4(&a4);
A a5(a5.zero());
A a6(a6.ONE);
A a7 = getA();
A a8 = getA(a8.TWO);
A a9 = getA(&a9);
A a10(a10.count);

A a11(a11);  // expected-warning {{variable 'a11' is uninitialized when used within its own initialization}}
A a12(a12.get());  // expected-warning {{variable 'a12' is uninitialized when used within its own initialization}}
A a13(a13.num);  // expected-warning {{variable 'a13' is uninitialized when used within its own initialization}}
A a14 = A(a14);  // expected-warning {{variable 'a14' is uninitialized when used within its own initialization}}
A a15 = getA(a15.num);  // expected-warning {{variable 'a15' is uninitialized when used within its own initialization}}
A a16(&a16.num);  // expected-warning {{variable 'a16' is uninitialized when used within its own initialization}}
A a17(a17.get2());  // expected-warning {{variable 'a17' is uninitialized when used within its own initialization}}
A a18 = x ? a18 : a17;  // expected-warning {{variable 'a18' is uninitialized when used within its own initialization}}
A a19 = getA(x ? a19 : a17);  // expected-warning {{variable 'a19' is uninitialized when used within its own initialization}}
A a20{a20};  // expected-warning {{variable 'a20' is uninitialized when used within its own initialization}}
A a21 = {a21};  // expected-warning {{variable 'a21' is uninitialized when used within its own initialization}}

A *a22 = new A(a22->count);
A *a23 = new A(a23->ONE);
A *a24 = new A(a24->TWO);
A *a25 = new A(a25->zero());

A *a26 = new A(a26->get());    // expected-warning {{variable 'a26' is uninitialized when used within its own initialization}}
A *a27 = new A(a27->get2());  // expected-warning {{variable 'a27' is uninitialized when used within its own initialization}}
A *a28 = new A(a28->num);  // expected-warning {{variable 'a28' is uninitialized when used within its own initialization}}

struct B {
  // POD struct.
  int x;
  int *y;
};

B getB() { return B(); };
B getB(int x) { return B(); };
B getB(int *x) { return B(); };
B getB(B *b) { return B(); };

B* getPtrB() { return 0; };
B* getPtrB(int x) { return 0; };
B* getPtrB(int *x) { return 0; };
B* getPtrB(B **b) { return 0; };

void setupB() {
  B b1;
  B b2(b1);
  B b3 = { 5, &b3.x };
  B b4 = getB();
  B b5 = getB(&b5);
  B b6 = getB(&b6.x);

  // Silence unused warning
  (void) b2;
  (void) b4;

  B b7(b7);  // expected-warning {{variable 'b7' is uninitialized when used within its own initialization}}
  B b8 = getB(b8.x);  // expected-warning {{variable 'b8' is uninitialized when used within its own initialization}}
  B b9 = getB(b9.y);  // expected-warning {{variable 'b9' is uninitialized when used within its own initialization}}
  B b10 = getB(-b10.x);  // expected-warning {{variable 'b10' is uninitialized when used within its own initialization}}

  B* b11 = 0;
  B* b12(b11);
  B* b13 = getPtrB();
  B* b14 = getPtrB(&b14);

  (void) b12;
  (void) b13;

  B* b15 = getPtrB(b15->x);  // expected-warning {{variable 'b15' is uninitialized when used within its own initialization}}
  B* b16 = getPtrB(b16->y);  // expected-warning {{variable 'b16' is uninitialized when used within its own initialization}}

  B b17 = { b17.x = 5, b17.y = 0 };
  B b18 = { b18.x + 1, b18.y };  // expected-warning 2{{variable 'b18' is uninitialized when used within its own initialization}}
}

B b1;
B b2(b1);
B b3 = { 5, &b3.x };
B b4 = getB();
B b5 = getB(&b5);
B b6 = getB(&b6.x);

B b7(b7);  // expected-warning {{variable 'b7' is uninitialized when used within its own initialization}}
B b8 = getB(b8.x);  // expected-warning {{variable 'b8' is uninitialized when used within its own initialization}}
B b9 = getB(b9.y);  // expected-warning {{variable 'b9' is uninitialized when used within its own initialization}}
B b10 = getB(-b10.x);  // expected-warning {{variable 'b10' is uninitialized when used within its own initialization}}

B* b11 = 0;
B* b12(b11);
B* b13 = getPtrB();
B* b14 = getPtrB(&b14);

B* b15 = getPtrB(b15->x);  // expected-warning {{variable 'b15' is uninitialized when used within its own initialization}}
B* b16 = getPtrB(b16->y);  // expected-warning {{variable 'b16' is uninitialized when used within its own initialization}}

B b17 = { b17.x = 5, b17.y = 0 };
B b18 = { b18.x + 1, b18.y };  // expected-warning 2{{variable 'b18' is uninitialized when used within its own initialization}}


// Also test similar constructs in a field's initializer.
struct S {
  int x;
  void *ptr;

  S(bool (*)[1]) : x(x) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(bool (*)[2]) : x(x + 1) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(bool (*)[3]) : x(x + x) {} // expected-warning 2{{field 'x' is uninitialized when used here}}
  S(bool (*)[4]) : x(static_cast<long>(x) + 1) {} // expected-warning {{field 'x' is uninitialized when used here}}
  S(bool (*)[5]) : x(foo(x)) {} // expected-warning {{field 'x' is uninitialized when used here}}

  // These don't actually require the value of x and so shouldn't warn.
  S(char (*)[1]) : x(sizeof(x)) {} // rdar://8610363
  S(char (*)[2]) : ptr(&ptr) {}
  S(char (*)[3]) : x(__alignof__(x)) {}
  S(char (*)[4]) : x(bar(&x)) {}
  S(char (*)[5]) : x(boo(x)) {}
  S(char (*)[6]) : x(far(x)) {}
};

struct C { char a[100], *e; } car = { .e = car.a };

// <rdar://problem/10398199>
namespace rdar10398199 {
  class FooBase { protected: ~FooBase() {} };
  class Foo : public FooBase {
  public:
    operator int&() const;
  };
  void stuff();
  template <typename T> class FooImpl : public Foo {
    T val;
  public:
    FooImpl(const T &x) : val(x) {}
    ~FooImpl() { stuff(); }
  };

  template <typename T> FooImpl<T> makeFoo(const T& x) {
    return FooImpl<T>(x);
  }

  void test() {
    const Foo &x = makeFoo(42);
    const int&y = makeFoo(42u);
    (void)x;
    (void)y;
  };
}

// PR 12325 - this was a false uninitialized value warning due to
// a broken CFG.
int pr12325(int params) {
  int x = ({
    while (false)
      ;
    int _v = params;
    if (false)
      ;
    _v; // no-warning
  });
  return x;
}

// Test lambda expressions with -Wuninitialized
int test_lambda() {
  auto f1 = [] (int x, int y) { int z; return x + y + z; }; // expected-warning{{variable 'z' is uninitialized when used here}} expected-note {{initialize the variable 'z' to silence this warning}}
  return f1(1, 2);
}

namespace {
  struct A {
    enum { A1 };
    static int A2() {return 5;}
    int A3;
    int A4() { return 5;}
  };

  struct B {
    A a;
  };

  struct C {
    C() {}
    C(int x) {}
    static A a;
    B b;
  };
  A C::a = A();

  // Accessing non-static members will give a warning.
  struct D {
    C c;
    D(char (*)[1]) : c(c.b.a.A1) {}
    D(char (*)[2]) : c(c.b.a.A2()) {}
    D(char (*)[3]) : c(c.b.a.A3) {}    // expected-warning {{field 'c' is uninitialized when used here}}
    D(char (*)[4]) : c(c.b.a.A4()) {}  // expected-warning {{field 'c' is uninitialized when used here}}

    // c::a is static, so it is already initialized
    D(char (*)[5]) : c(c.a.A1) {}
    D(char (*)[6]) : c(c.a.A2()) {}
    D(char (*)[7]) : c(c.a.A3) {}
    D(char (*)[8]) : c(c.a.A4()) {}
  };

  struct E {
    int a, b, c;
    E(char (*)[1]) : a(a ? b : c) {}  // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[2]) : a(b ? a : a) {} // expected-warning 2{{field 'a' is uninitialized when used here}}
    E(char (*)[3]) : a(b ? (a) : c) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[4]) : a(b ? c : (a+c)) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[5]) : a(b ? c : b) {}

    E(char (*)[6]) : a(a ?: a) {} // expected-warning 2{{field 'a' is uninitialized when used here}}
    E(char (*)[7]) : a(b ?: a) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[8]) : a(a ?: c) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[9]) : a(b ?: c) {}

    E(char (*)[10]) : a((a, a, b)) {}
    E(char (*)[11]) : a((c + a, a + 1, b)) {} // expected-warning 2{{field 'a' is uninitialized when used here}}
    E(char (*)[12]) : a((b + c, c, a)) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[13]) : a((a, a, a, a)) {} // expected-warning {{field 'a' is uninitialized when used here}}
    E(char (*)[14]) : a((b, c, c)) {}
  };

  struct F {
    int a;
    F* f;
    F(int) {}
    F() {}
  };

  int F::*ptr = &F::a;
  F* F::*f_ptr = &F::f;
  struct G {
    F f1, f2;
    F *f3, *f4;
    G(char (*)[1]) : f1(f1) {} // expected-warning {{field 'f1' is uninitialized when used here}}
    G(char (*)[2]) : f2(f1) {}
    G(char (*)[3]) : f2(F()) {}

    G(char (*)[4]) : f1(f1.*ptr) {} // expected-warning {{field 'f1' is uninitialized when used here}}
    G(char (*)[5]) : f2(f1.*ptr) {}

    G(char (*)[6]) : f3(f3) {}  // expected-warning {{field 'f3' is uninitialized when used here}}
    G(char (*)[7]) : f3(f3->*f_ptr) {} // expected-warning {{field 'f3' is uninitialized when used here}}
    G(char (*)[8]) : f3(new F(f3->*ptr)) {} // expected-warning {{field 'f3' is uninitialized when used here}}
  };
}

namespace statics {
  static int a = a; // no-warning: used to signal intended lack of initialization.
  static int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
  static int c = (c + c); // expected-warning 2{{variable 'c' is uninitialized when used within its own initialization}}
  static int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
  static int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

  // Thes don't warn as they don't require the value.
  static int g = sizeof(g);
  int gg = g;  // Silence unneeded warning
  static void* ptr = &ptr;
  static int h = bar(&h);
  static int i = boo(i);
  static int j = far(j);
  static int k = __alignof__(k);

  static int l = k ? l : l;  // expected-warning 2{{variable 'l' is uninitialized when used within its own initialization}}
  static int m = 1 + (k ? m : m);  // expected-warning 2{{variable 'm' is uninitialized when used within its own initialization}}
  static int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}

  void test() {
    static int a = a; // no-warning: used to signal intended lack of initialization.
    static int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
    static int c = (c + c); // expected-warning 2{{variable 'c' is uninitialized when used within its own initialization}}
    static int d = ({ d + d ;}); // expected-warning 2{{variable 'd' is uninitialized when used within its own initialization}}
    static int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
    static int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

    // Thes don't warn as they don't require the value.
    static int g = sizeof(g);
    static void* ptr = &ptr;
    static int h = bar(&h);
    static int i = boo(i);
    static int j = far(j);
    static int k = __alignof__(k);

    static int l = k ? l : l;  // expected-warning 2{{variable 'l' is uninitialized when used within its own initialization}}
    static int m = 1 + (k ? m : m);  // expected-warning 2{{variable 'm' is uninitialized when used within its own initialization}}
    static int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
   for (;;) {
      static int a = a; // no-warning: used to signal intended lack of initialization.
      static int b = b + 1; // expected-warning {{variable 'b' is uninitialized when used within its own initialization}}
      static int c = (c + c); // expected-warning 2{{variable 'c' is uninitialized when used within its own initialization}}
      static int d = ({ d + d ;}); // expected-warning 2{{variable 'd' is uninitialized when used within its own initialization}}
      static int e = static_cast<long>(e) + 1; // expected-warning {{variable 'e' is uninitialized when used within its own initialization}}
      static int f = foo(f); // expected-warning {{variable 'f' is uninitialized when used within its own initialization}}

      // Thes don't warn as they don't require the value.
      static int g = sizeof(g);
      static void* ptr = &ptr;
      static int h = bar(&h);
      static int i = boo(i);
      static int j = far(j);
      static int k = __alignof__(k);

      static int l = k ? l : l;  // expected-warning 2{{variable 'l' is uninitialized when used within its own initialization}}
      static int m = 1 + (k ? m : m); // expected-warning 2{{variable 'm' is uninitialized when used within its own initialization}}
      static int n = -n;  // expected-warning {{variable 'n' is uninitialized when used within its own initialization}}
    }
  }
}

namespace in_class_initializers {
  struct S {
    S() : a(a + 1) {} // expected-warning{{field 'a' is uninitialized when used here}}
    int a = 42; // Note: because a is in a member initializer list, this initialization is ignored.
  };

  struct T {
    T() : b(a + 1) {} // No-warning.
    int a = 42;
    int b;
  };

  struct U {
    U() : a(b + 1), b(a + 1) {} // FIXME: Warn here.
    int a = 42; // Note: because a and b are in the member initializer list, these initializers are ignored.
    int b = 1;
  };
}

namespace references {
  int &a = a; // expected-warning{{reference 'a' is not yet bound to a value when used within its own initialization}}
  int &b(b); // expected-warning{{reference 'b' is not yet bound to a value when used within its own initialization}}
  int &c = a ? b : c; // expected-warning{{reference 'c' is not yet bound to a value when used within its own initialization}}
  int &d{d}; // expected-warning{{reference 'd' is not yet bound to a value when used within its own initialization}}

  struct S {
    S() : a(a) {} // expected-warning{{reference 'a' is not yet bound to a value when used here}}
    int &a;
  };

  void f() {
    int &a = a; // expected-warning{{reference 'a' is not yet bound to a value when used within its own initialization}}
    int &b(b); // expected-warning{{reference 'b' is not yet bound to a value when used within its own initialization}}
    int &c = a ? b : c; // expected-warning{{reference 'c' is not yet bound to a value when used within its own initialization}}
    int &d{d}; // expected-warning{{reference 'd' is not yet bound to a value when used within its own initialization}}
  }

  struct T {
    T() : a(b), b(a) {} // FIXME: Warn here.
    int &a, &b;
    int &c = c; // expected-warning{{reference 'c' is not yet bound to a value when used here}}
  };

  int x;
  struct U {
    U() : b(a) {} // No-warning.
    int &a = x;
    int &b;
  };
}
