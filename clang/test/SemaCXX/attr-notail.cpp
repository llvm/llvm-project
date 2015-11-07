// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

class Base {
public:
  [[clang::not_tail_called]] virtual int foo1(); // expected-error {{'not_tail_called' attribute cannot be applied to virtual functions}}
  virtual int foo2();
  [[clang::not_tail_called]] int foo3();
  virtual ~Base() {}
};

class Derived1 : public Base {
public:
  int foo1() override;
  [[clang::not_tail_called]] int foo2() override; // expected-error {{'not_tail_called' attribute cannot be applied to virtual functions}}
  [[clang::not_tail_called]] int foo4();
};
