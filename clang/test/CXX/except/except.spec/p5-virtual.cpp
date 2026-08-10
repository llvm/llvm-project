// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// Compatibility of virtual functions.

struct A
{
};

struct B1 : A
{
};

struct B2 : A
{
};

struct D : B1, B2
{
};

struct P : private A
{
};

struct Base
{
  virtual void f1() throw();
  virtual void f2() throw(int, float);

  virtual void f3() throw(int, float);
  virtual void f4() throw(A);
  virtual void f5() throw(A, int, float);
  virtual void f6();

  virtual void f7() noexcept;
  virtual void f8() noexcept;
  virtual void f9() noexcept(false);
  virtual void f10() noexcept(false);

  virtual void f11() throw();
  virtual void f12() noexcept;
  virtual void f13() noexcept(false);
  virtual void f14() throw(int);

  virtual void f15();
  virtual void f16();

  virtual void g1() throw(); // expected-note {{overridden virtual function is here}}
  virtual void g2() throw(int); // expected-note {{overridden virtual function is here}}
  virtual void g3() throw(A); // expected-note {{overridden virtual function is here}}
  virtual void g4() throw(B1); // expected-note {{overridden virtual function is here}}
  virtual void g5() throw(A); // expected-note {{overridden virtual function is here}}

  virtual void g6() noexcept; // expected-note {{overridden virtual function is here}}
  virtual void g7() noexcept; // expected-note {{overridden virtual function is here}}

  virtual void g8() noexcept; // expected-note {{overridden virtual function is here}}
  virtual void g9() throw(); // expected-note {{overridden virtual function is here}}
  virtual void g10() throw(int); // expected-note {{overridden virtual function is here}}
};
struct Derived : Base
{
  virtual void f1() throw();
  virtual void f2() throw(float, int);

  virtual void f3() throw(float);
  virtual void f4() throw(B1);
  virtual void f5() throw(B1, B2, int);
  virtual void f6() throw(B2, B2, int, float, char, double, bool);

  virtual void f7() noexcept;
  virtual void f8() noexcept(true);
  virtual void f9() noexcept(true);
  virtual void f10() noexcept(false);

  virtual void f11() noexcept;
  virtual void f12() throw();
  virtual void f13() throw(int);
  virtual void f14() noexcept(true);

  virtual void f15() noexcept;
  virtual void f16() throw();

  virtual void g1() throw(int); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g2(); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g3() throw(D); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g4() throw(A); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g5() throw(P); // expected-error {{exception specification of overriding function is more lax}}

  virtual void g6() noexcept(false); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g7(); // expected-error {{exception specification of overriding function is more lax}}

  virtual void g8() throw(int); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g9() noexcept(false); // expected-error {{exception specification of overriding function is more lax}}
  virtual void g10() noexcept(false); // expected-error {{exception specification of overriding function is more lax}}
};
