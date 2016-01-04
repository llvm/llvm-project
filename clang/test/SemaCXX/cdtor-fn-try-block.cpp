// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify %s -std=c++14

int FileScope;

struct A {
  int I;
  void f();
  A() try {
  } catch (...) {
    I = 12; // expected-warning {{cannot refer to a non-static member from the handler of a constructor function try block}}
    f(); // expected-warning {{cannot refer to a non-static member from the handler of a constructor function try block}}

    FileScope = 12; // ok
    A a;
    a.I = 12; // ok
  }
};

struct B {
  int I;
  void f();
};

struct C : B {
  C() try {
  } catch (...) {
    I = 12; // expected-warning {{cannot refer to a non-static member from the handler of a constructor function try block}}
    f(); // expected-warning {{cannot refer to a non-static member from the handler of a constructor function try block}}
  }
};

struct D {
  static int I;
  static void f();

  D() try {
  } catch (...) {
    I = 12; // ok
    f(); // ok
  }
};
int D::I;

struct E {
  int I;
  void f();
  static int J;
  static void g();

  ~E() try {
  } catch (...) {
    I = 12; // expected-warning {{cannot refer to a non-static member from the handler of a destructor function try block}}
    f(); // expected-warning {{cannot refer to a non-static member from the handler of a destructor function try block}}

    J = 12; // ok
    g(); // ok
  }
};
int E::J;

struct F {
  static int I;
  static void f();
};
int F::I;

struct G : F {
  G() try {
  } catch (...) {
    I = 12; // ok
    f(); // ok
  }
};

struct H {
  struct A {};
  enum {
    E
  };

  H() try {
  } catch (...) {
    H::A a; // ok
    int I = E; // ok
  }
};

struct I {
  int J;

  I() {
    try { // not a function-try-block
    } catch (...) {
      J = 12; // ok
	}
  }
};