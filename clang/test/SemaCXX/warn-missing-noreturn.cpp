// RUN: %clang_cc1 -fsyntax-only -verify %s -Wmissing-noreturn -Wreturn-type
void f() __attribute__((noreturn));

template<typename T> void g(T) {
  f();
}

template void g<int>(int);

template<typename T> struct A {
  void g() {
    f();
  }
};

template struct A<int>;

struct B {
  template<typename T> void g(T) {
    f();
  }
};

template void B::g<int>(int);

// We don't want a warning here.
struct X {
  virtual void g() { f(); }
};

namespace test1 {
  bool condition();

  // We don't want a warning here.
  void foo() {
    while (condition()) {}
  }
}


// <rdar://problem/7880658> - This test case previously had a false "missing return"
// warning.
struct R7880658 {
  R7880658 &operator++();
  bool operator==(const R7880658 &) const;
  bool operator!=(const R7880658 &) const;
};

void f_R7880658(R7880658 f, R7880658 l) {  // no-warning
  for (; f != l; ++f) {
  }
}

namespace test2 {

  bool g();
  void *h() __attribute__((noreturn));
  void *j();

  struct A {
    void *f;

    A() : f(0) { }
    A(int) : f(h()) { } // expected-warning {{function 'A' could be declared with attribute 'noreturn'}}
    A(char) : f(j()) { }
    A(bool b) : f(b ? h() : j()) { }
  };
}

namespace test3 {
  struct A {
    ~A();
  };

  struct B {
    ~B() { }

    A a;
  };

  struct C : A { 
    ~C() { }
  };
}

// <rdar://problem/8875247> - Properly handle CFGs with destructors.
struct rdar8875247 {
  ~rdar8875247 ();
};
void rdar8875247_aux();

int rdar8875247_test() {
  rdar8875247 f;
} // expected-warning{{non-void function does not return a value}}

struct rdar8875247_B {
  rdar8875247_B();
  ~rdar8875247_B();
};

rdar8875247_B test_rdar8875247_B() {
  rdar8875247_B f;
  return f;
} // no-warning

namespace PR10801 {
  struct Foo {
    void wibble() __attribute((__noreturn__));
  };

  struct Bar {
    void wibble();
  };

  template <typename T> void thingy(T thing) {
    thing.wibble();
  }

  void test() {
    Foo f;
    Bar b;
    thingy(f);
    thingy(b);
  }
}
