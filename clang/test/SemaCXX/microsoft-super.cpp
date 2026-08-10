// RUN: %clang_cc1 -fms-extensions -verify %s

namespace test0 {
  class A {
  private:
    void foo(int*);
  public:
    void foo(long*);
  };
  class B : public A {
    void test() {
      __super::foo((long*) 0);
    }
  };
}

namespace test1 {
  struct A {
    static void foo(); // expected-note {{member is declared here}}
  };
  struct B : private A { // expected-note {{constrained by private inheritance here}}
    void test() {
      __super::foo();
    }
  };
  struct C : public B {
    void test() {
      __super::foo(); // expected-error {{'foo' is a private member of 'test1::A'}}
    }
  };
}

namespace test2 {
  struct A {
    static void foo();
  };
  struct B : public A {
    void test() {
      __super::foo();
    }
  };
  struct C : private B {
    void test() {
      __super::foo();
    }
  };
}
