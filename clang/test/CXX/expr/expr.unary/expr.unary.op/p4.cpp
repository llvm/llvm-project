// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  struct A {
    void foo(void (A::*)(int)); // expected-note {{passing argument to parameter here}}
    template<typename T> void g(T);

    void test() {
      foo(&g<int>); // expected-error-re {{cannot form member pointer of type 'void (test0::A::*)(int){{( __attribute__\(\(thiscall\)\))?}}' without '&' and class name}}
    }
  };
}

// This should succeed.
namespace test1 {
  struct A {
    static void f(void (A::*)());
    static void f(void (*)(int));
    void g();
    static void g(int);

    void test() {
      f(&g);
    }
  };
}

namespace test2 {
  struct A {
    static int foo(short);
    static int foo(float);
    int foo(int);
    int foo(double);

    void test();
  };

  void A::test() {
    // FIXME: The error message in this case is less than clear, we can do
    // better.
    int (A::*ptr)(int) = &(A::foo); // expected-error {{cannot create a non-constant pointer to member function}}
  }
}
