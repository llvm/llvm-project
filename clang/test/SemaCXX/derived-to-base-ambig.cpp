// RUN: %clang_cc1 -fsyntax-only -verify %s
class A { };
class B : public A { };
class C : public A { };
class D : public B, public C { };

void f(D* d) {
  A* a;
  a = d; // expected-error{{ambiguous conversion from derived class 'D' to base class 'A':}}
}

class Object2 { };
class A2 : public Object2 { };
class B2 : public virtual A2 { };
class C2 : virtual public A2 { };
class D2 : public B2, public C2 { };
class E2 : public D2, public C2, public virtual A2 { }; // expected-warning{{direct base 'C2' is inaccessible due to ambiguity:\n    class E2 -> class D2 -> class C2\n    class E2 -> class C2}}
class F2 : public E2, public A2 { }; // expected-warning{{direct base 'A2' is inaccessible due to ambiguity:\n    class F2 -> class E2 -> class D2 -> class B2 -> class A2\n    class F2 -> class A2}}

void g(E2* e2, F2* f2) {
  Object2* o2;
  o2 = e2;
  o2 = f2; // expected-error{{ambiguous conversion from derived class 'F2' to base class 'Object2':}}
}

// Test that ambiguous/inaccessibility checking does not trigger too
// early, because it should not apply during overload resolution.
void overload_okay(Object2*);
void overload_okay(E2*);

void overload_call(F2* f2) {
  overload_okay(f2);
}
