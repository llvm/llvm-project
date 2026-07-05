// RUN: %clang_cc1 -fsyntax-only -verify -fdiagnostics-parseable-fixits %s

class A {
  virtual void foo();
};
class B : public A {
  void foo() override;
};

void B::foo() override {} // expected-error {{'override' specifier is not allowed outside a class definition}}
                          // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:15-[[@LINE-1]]:24}:""

void f1() override; // expected-error {{'override' specifier is not allowed}}

void f2() override {} // expected-error {{'override' specifier is not allowed}}

void test() {
  void f() override; // expected-error {{'override' specifier is not allowed}}
}
