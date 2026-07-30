// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR12497
namespace test0 {
  class A {
  protected:
    A() {}
    A(const A &) {}
    ~A() {}
    A &operator=(const A &a) { return *this; }
  };

  class B : public A {};

  void test() {
    B b1;
    B b2 = b1;
    b1 = b2;
  }
}
