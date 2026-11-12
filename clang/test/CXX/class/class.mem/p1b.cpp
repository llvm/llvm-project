// The first run checks that the correct errors are generated,
// implicitly checking the order of default argument parsing:
// RUN: %clang_cc1 -fsyntax-only -verify %s
// The second run checks the order of inline method definitions:
// RUN: not %clang_cc1 -fsyntax-only %s 2> %t
// RUN: FileCheck %s < %t

class A {
public:
  void a1() {
    B b = B();
  }

  class B;
  void a2(B b = B()); // expected-error{{use of default argument to function 'B' that is declared later in class 'B'}}

  void a3(int a = 42);

  // CHECK: error: use of undeclared identifier 'first'
  void a4(int a = first); // expected-error{{use of undeclared identifier 'first'}}

  class B {
  public:
    B(int b = 42) { // expected-note{{default argument declared here}}
      A a;
      a.a3();
      a.a6();
    }

    void b1(A a = A()); // expected-error{{use of default argument to function 'A' that is declared later in class 'A'}}

    // CHECK: error: use of undeclared identifier 'second'
    void b2(int a = second); // expected-error{{use of undeclared identifier 'second'}}
  };

  void a5() {
    B b = B();
  }

  void a6(B b = B());

  A(int a = 42); // expected-note{{default argument declared here}}

  // CHECK: error: use of undeclared identifier 'third'
  void a7(int a = third); // expected-error{{use of undeclared identifier 'third'}}
};
