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

struct E;
struct F;
struct G;
struct H;
struct D {
  virtual E &operator=(const E &); // expected-note {{here}}
  virtual F &operator=(const F &);
  virtual G &operator=(G&&); // expected-note {{here}}
  virtual H &operator=(H&&); // expected-note {{here}}
  friend struct F;

private:
  D &operator=(const D&) = default;
  D &operator=(D&&) = default;
  virtual ~D(); // expected-note 2{{here}}
};
struct E : D {};
// expected-error@-1 {{deleted function '~E' cannot override a non-deleted function}}
// expected-note@-2 {{destructor of 'E' is implicitly deleted because base class 'D' has an inaccessible destructor}}
// expected-error@-3 {{deleted function 'operator=' cannot override a non-deleted function}}
// expected-note@-4 {{copy assignment operator of 'E' is implicitly deleted because base class 'D' has an inaccessible copy assignment operator}}
struct F : D {};
struct G : D {};
// expected-error@-1 {{deleted function '~G' cannot override a non-deleted function}}
// expected-note@-2 {{destructor of 'G' is implicitly deleted because base class 'D' has an inaccessible destructor}}
// expected-error@-3 {{deleted function 'operator=' cannot override a non-deleted function}}
// expected-note@-4 {{move assignment operator of 'G' is implicitly deleted because base class 'D' has an inaccessible move assignment operator}}
struct H : D { // expected-note {{deleted because base class 'D' has an inaccessible move assignment}}
  H &operator=(H&&) = default; // expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
  // expected-error@-1 {{deleted function 'operator=' cannot override a non-deleted function}}
  // expected-note@-3 {{move assignment operator of 'H' is implicitly deleted because base class 'D' has an inaccessible move assignment operator}}
  ~H();
};
