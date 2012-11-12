// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s 

namespace Test1 {

struct B {
  virtual void f(int);
};

struct D : B {
  virtual void f(long) override; // expected-error {{'f' marked 'override' but does not override any member functions}}
  void f(int) override;
};
}

namespace Test2 {

struct A {
  virtual void f(int, char, int);
};

template<typename T>
struct B : A {
  // FIXME: Diagnose this.
  virtual void f(T) override;
};

template<typename T>
struct C : A {
  virtual void f(int) override; // expected-error {{does not override}}
};

}

namespace Test3 {

struct A {
  virtual void f(int, char, int);
};

template<typename... Args>
struct B : A { 
  virtual void f(Args...) override; // expected-error {{'f' marked 'override' but does not override any member functions}}
};

template struct B<int, char, int>;
template struct B<int>; // expected-note {{in instantiation of template class 'Test3::B<int>' requested here}}

}

namespace Test4 {
struct B {
  virtual void f() const final; // expected-note {{overridden virtual function is here}}
};

struct D : B {
  void f() const; // expected-error {{declaration of 'f' overrides a 'final' function}}
};

}

namespace PR13499 {
  struct X {
    virtual void f();
    virtual void h();
  };
  template<typename T> struct A : X {
    void f() override;
    void h() final;
  };
  template<typename T> struct B : X {
    void g() override; // expected-error {{only virtual member functions can be marked 'override'}}
    void i() final; // expected-error {{only virtual member functions can be marked 'final'}}
  };
  B<int> b; // no-note
  template<typename T> struct C : T {
    void g() override;
    void i() final;
  };
  template<typename T> struct D : X {
    virtual void g() override; // expected-error {{does not override}}
    virtual void i() final;
  };
  template<typename...T> struct E : X {
    void f(T...) override;
    void g(T...) override; // expected-error {{only virtual member functions can be marked 'override'}}
    void h(T...) final;
    void i(T...) final; // expected-error {{only virtual member functions can be marked 'final'}}
  };
  // FIXME: Diagnose these in the template definition, not in the instantiation.
  E<> e; // expected-note {{in instantiation of}}

  template<typename T> struct Y : T {
    void f() override;
    void h() final;
  };
  template<typename T> struct Z : T {
    void g() override; // expected-error {{only virtual member functions can be marked 'override'}}
    void i() final; // expected-error {{only virtual member functions can be marked 'final'}}
  };
  Y<X> y;
  Z<X> z; // expected-note {{in instantiation of}}
}
