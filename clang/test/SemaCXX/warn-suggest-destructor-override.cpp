// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify -Wsuggest-destructor-override

struct A {
  ~A();
  virtual void run();
};

struct B : public A {
  ~B();
};

struct C {
  virtual void run();
  virtual ~C();  // expected-note 2{{overridden virtual function is here}}
};

struct D : public C {
  void run();
  ~D();
  // expected-warning@-1 {{'~D' overrides a destructor but is not marked 'override'}}
};

struct E : public C {
  void run();
  virtual ~E();
  // expected-warning@-1 {{'~E' overrides a destructor but is not marked 'override'}}
};
