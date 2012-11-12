// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

struct A {
  virtual void a(); // expected-note{{overridden virtual function is here}}
  virtual void b() = delete; // expected-note{{overridden virtual function is here}}
};

struct B: A {
  virtual void a() = delete; // expected-error{{deleted function 'a' cannot override a non-deleted function}}
  virtual void b(); // expected-error{{non-deleted function 'b' cannot override a deleted function}}
};

struct C: A {
  virtual void a();
  virtual void b() = delete;
};
