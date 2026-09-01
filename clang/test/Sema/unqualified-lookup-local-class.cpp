// RUN: %clang_cc1 -std=c++20 -verify %s

// Regression test for PR184622: out-of-line definitions of nested classes
// inside a function must not be visible via unqualified lookup.

// Unqualified member access inside a class body is fine.
struct A1 {
  int x;
  void f() { x = 1; } // OK
};

struct Base {
  int y;
};

struct Derived : Base {
  void f() { y = 2; } // OK - inherited member
};

struct A2 { int z; };
struct B2 : A2 {
  using A2::z;
  void f() { z = 3; } // OK
};

// Out-of-line definitions at namespace scope are fine.
class OuterNS { public: class Inner; };
class OuterNS::Inner {}; // OK - not inside a function

// Out-of-line definitions of types nested in a class inside a function:
// the qualified form must work, but the unqualified name must not leak.
void pass1() {
  struct A { struct B {}; };
  A::B b; // OK - qualified

  class C { public: class B; };
  class C::B {};
  C::B b2; // OK - qualified
}

template<class T>
struct Wrapper { Wrapper(T) {} };

// class nested in class, defined out-of-line inside function - unqualified 'B'
// must not be found.
void fail_class() {
  class A { public: class B; };
  class A::B {};
  B b;                      // expected-error {{unknown type name 'B'}}
  Wrapper w = Wrapper{B{}}; // expected-error {{use of undeclared identifier 'B'}}
  A::B ok;                  // OK - qualified lookup still works
}

// Same rule applies to structs (TagDecl covers class/struct/union/enum).
void fail_struct() {
  struct Outer { struct Inner; };
  struct Outer::Inner {};
  Inner bad;         // expected-error {{unknown type name 'Inner'}}
  Outer::Inner good; // OK
}

// Union nested in a class inside a function.
void fail_union() {
  struct S { union U; };
  union S::U { int a; float b; }; // expected-note {{'S::U' declared here}}
  U bad;   // expected-error {{unknown type name 'U'}}
  S::U ok; // OK
}

// Multiple independent functions: each has its own scope, so 'B' in fail2
// is unrelated to 'B' in fail_class above.
void fail2() {
  class Outer { public: class Inner; };
  class Outer::Inner {};
  Inner bad;         // expected-error {{unknown type name 'Inner'}}
  Outer::Inner good; // OK
}
