// RUN: %clang_cc1 -fsyntax-only -verify %s -Wnon-pod-varargs
class X { }; // expected-note {{the implicit copy constructor}} \
             // expected-note{{the implicit default constructor}}

int& copycon(X x); // expected-note{{passing argument to parameter}}
float& copycon(...);

void test_copycon(X x, X const xc, X volatile xv) {
  int& i1 = copycon(x);
  int& i2 = copycon(xc);
  copycon(xv); // expected-error{{no matching constructor}}
}

class A {
public:
  A(A&); // expected-note{{would lose const qualifier}} \
         // expected-note{{no known conversion}}
};

class B : public A { }; // expected-note{{would lose const qualifier}} \
// expected-note{{would lose volatile qualifier}} \
// expected-note 2{{requires 0 arguments}}

short& copycon2(A a); // expected-note{{passing argument to parameter}}
int& copycon2(B b); // expected-note 2{{passing argument to parameter}}
float& copycon2(...);

void test_copycon2(A a, const A ac, B b, B const bc, B volatile bv) {
  int& i1 = copycon2(b);
  copycon2(bc); // expected-error{{no matching constructor}}
  copycon2(bv); // expected-error{{no matching constructor}}
  short& s1 = copycon2(a);
  copycon2(ac); // expected-error{{no matching constructor}}
}

int& copycon3(A a); // expected-note{{passing argument to parameter 'a' here}}
float& copycon3(...);

void test_copycon3(B b, const B bc) {
  int& i1 = copycon3(b);
  copycon3(bc); // expected-error{{no matching constructor}}
}

class C : public B { };

float& copycon4(A a);
int& copycon4(B b);

void test_copycon4(C c) {
  int& i = copycon4(c);
};
