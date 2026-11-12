// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s

// Verify that using an initializer list for a non-aggregate looks for
// constructors..
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
  
  A(const A&) = delete; // expected-note 0-2{{'A' has been explicitly marked deleted here}}
};

struct B {
  A a;
};

struct C {
  const A& a;
};

void f() {
  A as1[1] = { };
  A as2[1] = { 1 };
#if __cplusplus <= 201402L
  // expected-error@-2 {{copying array element of type 'A' invokes deleted constructor}}
#endif

  B b1 = { };
  B b2 = { 1 };
#if __cplusplus <= 201402L
  // expected-error@-2 {{copying member subobject of type 'A' invokes deleted constructor}}
#endif
  
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

namespace IdiomaticStdArrayInitDoesNotWarn {
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wmissing-braces"
  template<typename T, int N> struct StdArray {
    T contents[N];
  };
  StdArray<int, 3> x = {1, 2, 3};
  
  template<typename T, int N> struct ArrayAndSomethingElse {
    T contents[N];
    int something_else;
  };
  ArrayAndSomethingElse<int, 3> y = {1, 2, 3}; // expected-warning {{suggest braces}}

#if __cplusplus >= 201703L
  template<typename T, int N> struct ArrayAndBaseClass : StdArray<int, 3> {
    T contents[N];
  };
  ArrayAndBaseClass<int, 3> z = {1, 2, 3}; // expected-warning {{suggest braces}}

  // This pattern is used for tagged aggregates and must not warn
  template<typename T, int N> struct JustABaseClass : StdArray<T, N> {};
  JustABaseClass<int, 3> w = {1, 2, 3};
  // but this should be also ok
  JustABaseClass<int, 3> v = {{1, 2, 3}};

  template <typename T, int N> struct OnionBaseClass : JustABaseClass<T, N> {};
  OnionBaseClass<int, 3> u = {1, 2, 3};
  OnionBaseClass<int, 3> t = {{{1, 2, 3}}};

  struct EmptyBase {};

  template <typename T, int N> struct AggregateAndEmpty : StdArray<T, N>, EmptyBase {};
  AggregateAndEmpty<int, 3> p = {1, 2, 3}; // expected-warning {{suggest braces}}
#endif

#pragma clang diagnostic pop
}

namespace HugeArraysUseArrayFiller {
  // All we're checking here is that initialization completes in a reasonable
  // amount of time.
  struct A { int n; int arr[1000 * 1000 * 1000]; } a = {1, {2}};
}

namespace ElementDestructor {
  // The destructor for each element of class type is potentially invoked
  // (15.4 [class.dtor]) from the context where the aggregate initialization
  // occurs. Produce a diagnostic if an element's destructor isn't accessible.

  class X { int f; ~X(); }; // expected-note {{implicitly declared private here}}
  struct Y { X x; };

  void test0() {
    auto *y = new Y {}; // expected-error {{temporary of type 'X' has private destructor}}
  }

  struct S0 { int f; ~S0() = delete; }; // expected-note 3 {{'~S0' has been explicitly marked deleted here}}
  struct S1 { S0 s0; int f; };

  S1 test1() {
    auto *t = new S1 { .f = 1 }; // expected-error {{attempt to use a deleted function}}
    return {2}; // expected-error {{attempt to use a deleted function}}
  }

  // Check if the type of an array element has a destructor.
  struct S2 { S0 a[4]; };

  void test2() {
    auto *t = new S2 {1,2,3,4}; // expected-error {{attempt to use a deleted function}}
  }

#if __cplusplus >= 201703L
  namespace BaseDestructor {
     struct S0 { int f; ~S0() = delete; }; // expected-note {{'~S0' has been explicitly marked deleted here}}

    // Check destructor of base class.
    struct S3 : S0 {};

    void test3() {
      S3 s3 = {1}; // expected-error {{attempt to use a deleted function}}
    }
  }
#endif

  // A's destructor doesn't have to be accessible from the context of C's
  // initialization.
  struct A { friend struct B; private: ~A(); };
  struct B { B(); A a; };
  struct C { B b; };
  C c = { B() };
}
