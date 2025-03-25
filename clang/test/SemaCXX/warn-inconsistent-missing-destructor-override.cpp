// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify -Winconsistent-missing-destructor-override

class A {
 public:
  ~A() {}
  void virtual run() {}
};

class B : public A {
 public:
  void run() override {}
  ~B() {}
};

class C {
 public:
  virtual void run() {}
  virtual ~C() {}  // expected-note 2{{overridden virtual function is here}}
};

class D : public C {
 public:
  void run() override {}
  ~D() {}
  // expected-warning@-1 {{'~D' overrides a destructor but is not marked 'override'}}
};

class E : public C {
 public:
  void run() override {}
  virtual ~E() {}
  // expected-warning@-1 {{'~E' overrides a destructor but is not marked 'override'}}
};
