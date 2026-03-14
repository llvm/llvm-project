// RUN: %clang_cc1 -fsyntax-only -verify %s 

struct A {
   A() : value(), cvalue() { } // expected-error {{reference to type 'int' requires an initializer}}
   int &value;
   const int cvalue;
};

struct B {
  int field;
};

struct X {
   X() { }      // expected-error {{constructor for 'X' must explicitly initialize the reference member 'value'}} \
                // expected-error {{constructor for 'X' must explicitly initialize the const member 'cvalue'}} \
                // expected-error {{constructor for 'X' must explicitly initialize the reference member 'b'}} \
                // expected-error {{constructor for 'X' must explicitly initialize the const member 'cb'}}
   int &value; // expected-note{{declared here}}
   const int cvalue; // expected-note{{declared here}}
   B& b; // expected-note{{declared here}}
   const B cb; // expected-note{{declared here}}
};


// PR5924
struct bar {};
bar xxx();

struct foo {
  foo_t a;  // expected-error {{unknown type name 'foo_t'}}
  foo() : a(xxx()) {}  // no error here.
};
