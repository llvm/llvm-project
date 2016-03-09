// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s 
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s 
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s 

// Verify that using an initializer list for a non-aggregate looks for
// constructors..
// Note that due to a (likely) standard bug, this is technically an aggregate,
// but we do not treat it as one.
struct NonAggr1 { // expected-note 2 {{candidate constructor}}
  NonAggr1(int, int) { } // expected-note {{candidate constructor}}

  int m;
};

struct Base { };
struct NonAggr2 : public Base { // expected-note 0-3 {{candidate constructor}}
  int m;
};

class NonAggr3 { // expected-note 3 {{candidate constructor}}
  int m;
};

struct NonAggr4 { // expected-note 3 {{candidate constructor}}
  int m;
  virtual void f();
};

NonAggr1 na1 = { 17 }; // expected-error{{no matching constructor for initialization of 'NonAggr1'}}
NonAggr2 na2 = { 17 };
NonAggr3 na3 = { 17 }; // expected-error{{no matching constructor for initialization of 'NonAggr3'}}
NonAggr4 na4 = { 17 }; // expected-error{{no matching constructor for initialization of 'NonAggr4'}}
#if __cplusplus <= 201402L
// expected-error@-4{{no matching constructor for initialization of 'NonAggr2'}}
#else
// expected-error@-6{{requires explicit braces}}
NonAggr2 na2b = { {}, 17 }; // ok
#endif

// PR5817
typedef int type[][2];
const type foo = {0};

// Vector initialization.
typedef short __v4hi __attribute__ ((__vector_size__ (8)));
__v4hi v1 = { (void *)1, 2, 3 }; // expected-error {{cannot initialize a vector element of type 'short' with an rvalue of type 'void *'}}

// Array initialization.
int a[] = { (void *)1 }; // expected-error {{cannot initialize an array element of type 'int' with an rvalue of type 'void *'}}

// Struct initialization.
struct S { int a; } s = { (void *)1 }; // expected-error {{cannot initialize a member subobject of type 'int' with an rvalue of type 'void *'}}

// Check that we're copy-initializing the structs.
struct A {
  A();
  A(int);
  ~A();
  
  A(const A&) = delete; // expected-note 2 {{'A' has been explicitly marked deleted here}}
};

struct B {
  A a;
};

struct C {
  const A& a;
};

void f() {
  A as1[1] = { };
  A as2[1] = { 1 }; // expected-error {{copying array element of type 'A' invokes deleted constructor}}

  B b1 = { };
  B b2 = { 1 }; // expected-error {{copying member subobject of type 'A' invokes deleted constructor}}
  
  C c1 = { 1 };
}

class Agg {
public:
  int i, j;
};

class AggAgg {
public:
  Agg agg1;
  Agg agg2;
};

AggAgg aggagg = { 1, 2, 3, 4 };

namespace diff_cpp14_dcl_init_aggr_example {
  struct derived;
  struct base {
    friend struct derived;
  private:
    base();
  };
  struct derived : base {};

  derived d1{};
#if __cplusplus > 201402L
  // expected-error@-2 {{private}}
  // expected-note@-7 {{here}}
#endif
  derived d2;
}

namespace ProtectedBaseCtor {
  // FIXME: It's unclear whether f() and g() should be valid in C++1z. What is
  // the object expression in a constructor call -- the base class subobject or
  // the complete object?
  struct A {
  protected:
    A();
  };

  struct B : public A {
    friend B f();
    friend B g();
    friend B h();
  };

  B f() { return {}; }
#if __cplusplus > 201402L
  // expected-error@-2 {{protected default constructor}}
  // expected-note@-12 {{here}}
#endif

  B g() { return {{}}; }
#if __cplusplus <= 201402L
  // expected-error@-2 {{no matching constructor}}
  // expected-note@-15 3{{candidate}}
#else
  // expected-error@-5 {{protected default constructor}}
  // expected-note@-21 {{here}}
#endif

  B h() { return {A{}}; }
#if __cplusplus <= 201402L
  // expected-error@-2 {{no matching constructor}}
  // expected-note@-24 3{{candidate}}
#endif
  // expected-error@-5 {{protected constructor}}
  // expected-note@-30 {{here}}
}
