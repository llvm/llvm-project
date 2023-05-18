// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {
  virtual void f() = 0; // expected-note{{unimplemented pure virtual method}}
};

struct B : A {
  virtual void f();
};

struct C : B {
  virtual void f() = 0; // expected-note 2{{unimplemented pure virtual method}}
};

struct D : C {
};

void test() {
  (void)new A; // expected-error{{abstract class}}
  (void)new B;
  (void)new C; // expected-error{{abstract class}}
  (void)new D; // expected-error{{abstract class}}
}
