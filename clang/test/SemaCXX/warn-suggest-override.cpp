// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify -Wsuggest-override

struct A {
  ~A();
  void run();
};

struct B : public A {
  ~B();
  void run();
};

struct C {
  virtual void run(); // expected-note 2{{overridden virtual function is here}}
  virtual ~C();
};

struct D : public C {
  void run();
  // expected-warning@-1 {{'run' overrides a member function but is not marked 'override'}}
  ~D();
};

struct E : public C {
  virtual void run();
  // expected-warning@-1 {{'run' overrides a member function but is not marked 'override'}}
  virtual ~E();
};

struct F : public C {
  void run() override;
  ~F() override;
};

struct G : public C { // expected-note {{mark 'G' as 'final'}}
  void run() final;
  ~G() final;
  // expected-warning@-1 {{class with destructor marked 'final' cannot be inherited from}}
};
