// RUN: %clang_cc1 -fsyntax-only -verify %s -Wshadow-all

// Basic cases, ambiguous paths, and fields with different access
class A {
public:
  int x;  // expected-note 2{{declared here}}
protected:
  int y;  // expected-note 2{{declared here}}
private:
  int z;
};

struct B : A {
};

struct C : A {
};

struct W {
  int w;  // expected-note {{declared here}}
};

struct U : W {
};

struct V : W {
};

class D {
public:
  char w; // expected-note {{declared here}}
private:
  char x;
};

// Check direct inheritance and multiple paths to the same base.
class E : B, C, D, U, V
{
  unsigned x;  // expected-warning {{non-static data member 'x' of 'E' shadows member inherited from type 'A'}}
  char y;  // expected-warning {{non-static data member 'y' of 'E' shadows member inherited from type 'A'}}
  double z;
  char w;  // expected-warning {{non-static data member 'w' of 'E' shadows member inherited from type 'D'}}  expected-warning {{non-static data member 'w' of 'E' shadows member inherited from type 'W'}}
};

// Virtual inheritance
struct F : virtual A {
};

struct G : virtual A {
};

class H : F, G {
  int x;  // expected-warning {{non-static data member 'x' of 'H' shadows member inherited from type 'A'}}
  int y;  // expected-warning {{non-static data member 'y' of 'H' shadows member inherited from type 'A'}}
  int z;
};

// Indirect inheritance
struct I {
  union {
    int x;  // expected-note {{declared here}}
    int y;
  };
};

struct J : I {
  int x;  // expected-warning {{non-static data member 'x' of 'J' shadows member inherited from type 'I'}}
};

// non-access paths
class N : W {
};

struct K {
  int y;
};

struct L : private K {
};

struct M : L {
  int y;
  int w;
};

// Multiple ambiguous paths with different accesses
struct A1 {
  int x;  // expected-note {{declared here}}
};

class B1 : A1 {
};

struct B2 : A1 {
};

struct C1 : B1, B2 {
};

class D1 : C1 {
};

struct D2 : C1 {
};

class D3 : C1 {
};

struct E1 : D1, D2, D3{
  int x;  // expected-warning {{non-static data member 'x' of 'E1' shadows member inherited from type 'A1'}}
};



