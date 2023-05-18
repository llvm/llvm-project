// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fsyntax-only -fixit -x c++ %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x c++ %t

namespace dcl_fct_default_p10 {
struct A {
  virtual void f(int a = 7); // expected-note{{'A::f' declared here}}
};

struct B : public A {
  void f(int a);
};

void m() {
  B* pb = new B;
  A* pa = pb;
  pa->f(); // OK, calls pa->B::f(7)
  pb->f(); // expected-error{{too few arguments to function call, expected 1, have 0; did you mean 'A::f'?}}
}
}

namespace PR18608 {
struct A {
virtual void f() const;
virtual void f(int x) const;  // expected-note{{'A::f' declared here}}
};

struct B : public A {
virtual void f() const;
};

void test(B b) {
  b.f(1);  // expected-error{{too many arguments to function call, expected 0, have 1; did you mean 'A::f'?}}
}
}

namespace PR20626 {
class A {
public:
  void Foo(){};  // expected-note{{'Foo' declared here}}
};
class B {};
class C : public A, public B {
  void Run() {
    B::Foo();  // expected-error{{no member named 'Foo' in 'PR20626::B'; did you mean simply 'Foo'?}}
  }
};
}
