// RUN: %clang_cc1 -std=c++11 -verify %s -fexceptions -fcxx-exceptions

struct A { struct X { virtual ~X() throw(Y); }; struct Y : X {}; };
struct B { struct X { virtual void f() throw(Y); }; struct Y : X { void f() throw(Y); }; };
struct C { struct X { virtual void f() throw(Y); }; struct Y : X { void f() throw(); }; };
struct D { struct X { virtual void f() throw(Y); }; struct Y : X { void f() noexcept; }; };
struct E { struct Y; struct X { virtual Y &operator=(const Y&) throw(Y); }; struct Y : X {}; };
struct F {
  struct X {
    virtual void f() throw(Y); // expected-note {{here}}
  };
  struct Y : X {
    void f() throw(int); // expected-error {{more lax}}
  };
};
