// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  struct A {
    static int foo;
  };
  
  namespace i0 {
    typedef int A; // expected-note {{declared here}}

    int test() {
      struct A a; // expected-error {{elaborated type refers to a typedef}}
      return a.foo;
    }
  }

  namespace i1 {
    template <class> class A; // expected-note {{declared here}}

    int test() {
      struct A a; // expected-error {{elaborated type refers to a template}}
      return a.foo;
    }
  }

  namespace i2 {
    int A;

    int test() {
      struct A a;
      return a.foo;
    }
  }

  namespace i3 {
    void A();

    int test() {
      struct A a;
      return a.foo;
    }
  }

  namespace i4 {
    template <class T> void A();

    int test() {
      struct A a;
      return a.foo;
    }
  }

  // This should magically be okay;  see comment in SemaDecl.cpp.
  // rdar://problem/7898108
  typedef struct A A;
  int test() {
    struct A a;
    return a.foo;
  }
}
