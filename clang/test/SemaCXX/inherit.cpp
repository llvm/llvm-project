// RUN: %clang_cc1 -fsyntax-only -verify %s
class A { };

class B1 : A { };

class B2 : virtual A { };

class B3 : virtual virtual A { }; // expected-error{{duplicate 'virtual' in base specifier}}

class C : public B1, private B2 { };


class D;                // expected-note {{forward declaration of 'D'}}

class E : public D { }; // expected-error{{base class has incomplete type}}

typedef int I;

class F : public I { }; // expected-error{{base specifier must name a class}}

union U1 : public A { }; // expected-error{{unions cannot have base classes}}

union U2 {};

class G : public U2 { }; // expected-error{{unions cannot be base classes}}

typedef G G_copy;
typedef G G_copy_2;
typedef G_copy G_copy_3;

class H : G_copy, A, G_copy_2, // expected-error{{base class 'G_copy' (aka 'G') specified more than once as a direct base class}}
          public G_copy_3 { }; // expected-error{{base class 'G_copy' (aka 'G') specified more than once as a direct base class}}
