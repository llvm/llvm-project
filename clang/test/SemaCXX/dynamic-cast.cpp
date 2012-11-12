// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {};
struct B : A {};
struct C : B {};

struct D : private A {};
struct E : A {};
struct F : B, E {};

struct Incomplete; // expected-note 2 {{forward declaration of 'Incomplete'}}

struct Poly
{
  virtual void f();
};

struct PolyDerived : Poly
{
};

void basic_bad()
{
  // ptr -> nonptr
  (void)dynamic_cast<A>((A*)0); // expected-error {{'A' is not a reference or pointer}}
  // nonptr -> ptr
  (void)dynamic_cast<A*>(0); // expected-error {{'int' is not a pointer}}
  // ptr -> noncls
  (void)dynamic_cast<int*>((A*)0); // expected-error {{'int' is not a class}}
  // noncls -> ptr
  (void)dynamic_cast<A*>((int*)0); // expected-error {{'int' is not a class}}
  // ref -> noncls
  (void)dynamic_cast<int&>(*((A*)0)); // expected-error {{'int' is not a class}}
  // noncls -> ref
  (void)dynamic_cast<A&>(*((int*)0)); // expected-error {{'int' is not a class}}
  // ptr -> incomplete
  (void)dynamic_cast<Incomplete*>((A*)0); // expected-error {{'Incomplete' is an incomplete type}}
  // incomplete -> ptr
  (void)dynamic_cast<A*>((Incomplete*)0); // expected-error {{'Incomplete' is an incomplete type}}
}

void same()
{
  (void)dynamic_cast<A*>((A*)0);
  (void)dynamic_cast<A&>(*((A*)0));
}

void up()
{
  (void)dynamic_cast<A*>((B*)0);
  (void)dynamic_cast<A&>(*((B*)0));
  (void)dynamic_cast<A*>((C*)0);
  (void)dynamic_cast<A&>(*((C*)0));

  // Inaccessible
  //(void)dynamic_cast<A*>((D*)0);
  //(void)dynamic_cast<A&>(*((D*)0));

  // Ambiguous
  (void)dynamic_cast<A*>((F*)0); // expected-error {{ambiguous conversion from derived class 'F' to base class 'A':\n    struct F -> struct B -> struct A\n    struct F -> struct E -> struct A}}
  (void)dynamic_cast<A&>(*((F*)0)); // expected-error {{ambiguous conversion from derived class 'F' to base class 'A':\n    struct F -> struct B -> struct A\n    struct F -> struct E -> struct A}}
}

void poly()
{
  (void)dynamic_cast<A*>((Poly*)0);
  (void)dynamic_cast<A&>(*((Poly*)0));
  (void)dynamic_cast<A*>((PolyDerived*)0);
  (void)dynamic_cast<A&>(*((PolyDerived*)0));

  // Not polymorphic source
  (void)dynamic_cast<Poly*>((A*)0); // expected-error {{'A' is not polymorphic}}
  (void)dynamic_cast<PolyDerived&>(*((A*)0)); // expected-error {{'A' is not polymorphic}}
}
