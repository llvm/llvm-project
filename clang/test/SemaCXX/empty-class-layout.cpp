// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify -Wno-inaccessible-base
// expected-no-diagnostics

#define SA(n, p) int a##n[(p) ? 1 : -1]

namespace Test0 {

struct A { int a; };
SA(0, sizeof(A) == 4);

struct B { };
SA(1, sizeof(B) == 1);

struct C : A, B { };
SA(2, sizeof(C) == 4);

struct D { };
struct E : D { };
struct F : E { };

struct G : E, F { };
SA(3, sizeof(G) == 2);

struct Empty { Empty(); };

struct I : Empty { 
  Empty e;
};
SA(4, sizeof(I) == 2);

struct J : Empty { 
  Empty e[2];
};
SA(5, sizeof(J) == 3);

template<int N> struct Derived : Empty, Derived<N - 1> { 
};
template<> struct Derived<0> : Empty { };

struct S1 : virtual Derived<10> { 
  Empty e;
};
SA(6, sizeof(S1) == 24);

struct S2 : virtual Derived<10> { 
  Empty e[2];
};
SA(7, sizeof(S2) == 24);

struct S3 {
  Empty e;
};

struct S4 : Empty, S3 { 
};
SA(8, sizeof(S4) == 2);

struct S5 : S3, Empty {};
SA(9, sizeof(S5) == 2);

struct S6 : S5 { };
SA(10, sizeof(S6) == 2);

struct S7 : Empty {
  void *v;
};
SA(11, sizeof(S7) == 8);

struct S8 : Empty, A {
};
SA(12, sizeof(S8) == 4);

}

namespace Test1 {

// Test that we don't try to place both A subobjects at offset 0.
struct A { };
class B { virtual void f(); };
class C : A, virtual B { };
struct D : virtual C { };
struct E : virtual A { };
class F : D, E { };

SA(0, sizeof(F) == 24);

}

namespace Test2 {

// Test that B::a isn't laid out at offset 0.
struct Empty { };
struct A : Empty { };
struct B : Empty {
  A a;
};

SA(0, sizeof(B) == 2);

}

namespace Test3 {

// Test that B::a isn't laid out at offset 0.
struct Empty { };
struct A { Empty e; };
struct B : Empty { A a; };
SA(0, sizeof(B) == 2);

}

namespace Test4 {

// Test that C::Empty isn't laid out at offset 0.
struct Empty { };
struct A : Empty { };
struct B { A a; };
struct C : B, Empty { };
SA(0, sizeof(C) == 2);

}

namespace Test5 {

// Test that B::Empty isn't laid out at offset 0.
struct Empty { };
struct Field : virtual Empty { };
struct A {
  Field f;
};
struct B : A, Empty { };
SA(0, sizeof(B) == 16);

}

namespace Test6 {

// Test that B::A isn't laid out at offset 0.
struct Empty { };
struct Field : virtual Empty { };
struct A {
  Field f;
};
struct B : Empty, A { };
SA(0, sizeof(B) == 16);

}

namespace Test7 {
  // Make sure we reserve enough space for both bases; PR11745.
  struct Empty { };
  struct Base1 : Empty { };
  struct Base2 : Empty { };
  struct Test : Base1, Base2 {
    char c;
  };
  SA(0, sizeof(Test) == 2);
}

namespace Test8 {
  // Test that type sugar doesn't make us incorrectly determine the size of an
  // array of empty classes.
  struct Empty1 {};
  struct Empty2 {};
  struct Empties : Empty1, Empty2 {};
  typedef Empty1 Sugar[4];
  struct A : Empty2, Empties {
    // This must go at offset 2, because if it were at offset 0,
    // V[0][1] would overlap Empties::Empty1.
    Sugar V[1];
  };
  SA(0, sizeof(A) == 6);
}
