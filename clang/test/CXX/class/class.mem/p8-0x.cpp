// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s 

struct Base1 { 
  virtual void g();
};

struct A : Base1 {
  virtual void g() override override; // expected-error {{class member already marked 'override'}}
  virtual void h() final final; // expected-error {{class member already marked 'final'}}
};

struct Base2 { 
  virtual void e1(), e2();
  virtual void f();
};

struct B : Base2 {
  virtual void e1() override, e2(int);  // No error.
  virtual void f() override;
  void g() override; // expected-error {{only virtual member functions can be marked 'override'}}
  int h override; // expected-error {{only virtual member functions can be marked 'override'}}
};

struct C {
  virtual void f() final;
  void g() final; // expected-error {{only virtual member functions can be marked 'final'}}
  int h final; // expected-error {{only virtual member functions can be marked 'final'}}
};

namespace inline_extension {
  struct Base1 { 
    virtual void g() {}
  };

  struct A : Base1 {
    virtual void g() override override {} // expected-error {{class member already marked 'override'}}
    virtual void h() final final {} // expected-error {{class member already marked 'final'}}
  };

  struct Base2 { 
    virtual void f();
  };

  struct B : Base2 {
    virtual void f() override {}
    void g() override {} // expected-error {{only virtual member functions can be marked 'override'}}
  };

  struct C {
    virtual void f() final {}
    void g() final {} // expected-error {{only virtual member functions can be marked 'final'}}
  };
}
