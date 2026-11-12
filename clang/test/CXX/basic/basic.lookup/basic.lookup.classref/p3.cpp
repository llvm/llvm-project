// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// C++0x [basic.lookup.classref]p3:
//   If the unqualified-id is ~type-name, the type-name is looked up in the 
//   context of the entire postfix-expression. If the type T of the object 
//   expression is of a class type C, the type-name is also looked up in the 
//   scope of class C. At least one of the lookups shall find a name that 
//   refers to (possibly cv-qualified) T.

// From core issue 305
struct A {
};

struct C {
  struct A {};
  void f ();
};

void C::f () {
  ::A *a;
  a->~A ();
}

// From core issue 414
struct X {};
void f() {
  X x;
  struct X {};
  x.~X();
}
