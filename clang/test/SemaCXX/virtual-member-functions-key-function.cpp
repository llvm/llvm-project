// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A {
  virtual ~A();
};

struct B : A {  // expected-error {{no suitable member 'operator delete' in 'B'}}
  B() { } // expected-note {{implicit default destructor for 'B' first required here}}
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
}; 

struct C : A {  // expected-error {{no suitable member 'operator delete' in 'C'}}
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
}; 

void f() {
  (void)new B; 
  (void)new C; // expected-note {{implicit default destructor for 'C' first required here}}
}

// Make sure that the key-function computation is consistent when the
// first virtual member function of a nested class has an inline body.
struct Outer {
  struct Inner {
    virtual void f() { }
    void g();
  };
};

void Outer::Inner::g() { }
