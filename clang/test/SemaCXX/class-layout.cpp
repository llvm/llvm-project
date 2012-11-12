// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify 
// expected-no-diagnostics

#define SA(n, p) int a##n[(p) ? 1 : -1]

struct A {
  int a;
  char b;
};

SA(0, sizeof(A) == 8);

struct B : A {
  char c;
};

SA(1, sizeof(B) == 12);

struct C {
// Make fields private so C won't be a POD type.
private:
  int a;
  char b;
};

SA(2, sizeof(C) == 8);

struct D : C {
  char c;
};

SA(3, sizeof(D) == 8);

struct __attribute__((packed)) E {
  char b;
  int a;
};

SA(4, sizeof(E) == 5);

struct __attribute__((packed)) F : E {
  char d;
};

SA(5, sizeof(F) == 6);

struct G { G(); };
struct H : G { };

SA(6, sizeof(H) == 1);

struct I {
  char b;
  int a;
} __attribute__((packed));

SA(6_1, sizeof(I) == 5);

// PR5580
namespace PR5580 {

class A { bool iv0 : 1; };
SA(7, sizeof(A) == 1);  

class B : A { bool iv0 : 1; };
SA(8, sizeof(B) == 2);

struct C { bool iv0 : 1; };
SA(9, sizeof(C) == 1);  

struct D : C { bool iv0 : 1; };
SA(10, sizeof(D) == 2);

}

namespace Test1 {

// Test that we don't assert on this hierarchy.
struct A { };
struct B : A { virtual void b(); };
class C : virtual A { int c; };
struct D : virtual B { };
struct E : C, virtual D { };
class F : virtual E { };
struct G : virtual E, F { };

SA(0, sizeof(G) == 24);

}

namespace Test2 {

// Test that this somewhat complex class structure is laid out correctly.
struct A { };
struct B : A { virtual void b(); };
struct C : virtual B { };
struct D : virtual A { };
struct E : virtual B, D { };
struct F : E, virtual C { };
struct G : virtual F, A { };
struct H { G g; };

SA(0, sizeof(H) == 24);

}
