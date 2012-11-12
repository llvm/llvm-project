// RUN: %clang_cc1 -fsyntax-only -verify %s
// REQUIRES: LP64

struct A {};

// ----------- const_cast --------------

typedef char c;
typedef c *cp;
typedef cp *cpp;
typedef cpp *cppp;
typedef cppp &cpppr;
typedef const cppp &cpppcr;
typedef const char cc;
typedef cc *ccp;
typedef volatile ccp ccvp;
typedef ccvp *ccvpp;
typedef const volatile ccvpp ccvpcvp;
typedef ccvpcvp *ccvpcvpp;
typedef int iar[100];
typedef iar &iarr;
typedef int (*f)(int);

void t_cc()
{
  ccvpcvpp var = 0;
  // Cast away deep consts and volatiles.
  char ***var2 = (cppp)(var);
  char ***const &var3 = var2;
  // Const reference to reference.
  char ***&var4 = (cpppr)(var3);
  // Drop reference. Intentionally without qualifier change.
  char *** var5 = (cppp)(var4);
  const int ar[100] = {0};
  // Array decay. Intentionally without qualifier change.
  int *pi = (int*)(ar);
  f fp = 0;
  // Don't misidentify fn** as a function pointer.
  f *fpp = (f*)(&fp);
  int const A::* const A::*icapcap = 0;
  int A::* A::* iapap = (int A::* A::*)(icapcap);
}

// ----------- static_cast -------------

struct B : public A {};             // Single public base.
struct C1 : public virtual B {};    // Single virtual base.
struct C2 : public virtual B {};
struct D : public C1, public C2 {}; // Diamond
struct E : private A {};            // Single private base.
struct F : public C1 {};            // Single path to B with virtual.
struct G1 : public B {};
struct G2 : public B {};
struct H : public G1, public G2 {}; // Ambiguous path to B.

enum Enum { En1, En2 };
enum Onom { On1, On2 };

struct Co1 { operator int(); };
struct Co2 { Co2(int); };
struct Co3 { };
struct Co4 { Co4(Co3); operator Co3(); };

// Explicit implicits
void t_529_2()
{
  int i = 1;
  (void)(float)(i);
  double d = 1.0;
  (void)(float)(d);
  (void)(int)(d);
  (void)(char)(i);
  (void)(unsigned long)(i);
  (void)(int)(En1);
  (void)(double)(En1);
  (void)(int&)(i);
  (void)(const int&)(i);

  int ar[1];
  (void)(const int*)(ar);
  (void)(void (*)())(t_529_2);

  (void)(void*)(0);
  (void)(void*)((int*)0);
  (void)(volatile const void*)((const int*)0);
  (void)(A*)((B*)0);
  (void)(A&)(*((B*)0));
  (void)(const B*)((C1*)0);
  (void)(B&)(*((C1*)0));
  (void)(A*)((D*)0);
  (void)(const A&)(*((D*)0));
  (void)(int B::*)((int A::*)0);
  (void)(void (B::*)())((void (A::*)())0);
  (void)(A*)((E*)0); // C-style cast ignores access control
  (void)(void*)((const int*)0); // const_cast appended

  (void)(int)(Co1());
  (void)(Co2)(1);
  (void)(Co3)((Co4)(Co3()));

  // Bad code below
  //(void)(A*)((H*)0); // {{static_cast from 'struct H *' to 'struct A *' is not allowed}}
}

// Anything to void
void t_529_4()
{
  (void)(1);
  (void)(t_529_4);
}

// Static downcasts
void t_529_5_8()
{
  (void)(B*)((A*)0);
  (void)(B&)(*((A*)0));
  (void)(const G1*)((A*)0);
  (void)(const G1&)(*((A*)0));
  (void)(B*)((const A*)0); // const_cast appended
  (void)(B&)(*((const A*)0)); // const_cast appended
  (void)(E*)((A*)0); // access control ignored
  (void)(E&)(*((A*)0)); // access control ignored

  // Bad code below

  (void)(C1*)((A*)0); // expected-error {{cannot cast 'A *' to 'C1 *' via virtual base 'B'}}
  (void)(C1&)(*((A*)0)); // expected-error {{cannot cast 'A' to 'C1 &' via virtual base 'B'}}
  (void)(D*)((A*)0); // expected-error {{cannot cast 'A *' to 'D *' via virtual base 'B'}}
  (void)(D&)(*((A*)0)); // expected-error {{cannot cast 'A' to 'D &' via virtual base 'B'}}
  (void)(H*)((A*)0); // expected-error {{ambiguous cast from base 'A' to derived 'H':\n    struct A -> struct B -> struct G1 -> struct H\n    struct A -> struct B -> struct G2 -> struct H}}
  (void)(H&)(*((A*)0)); // expected-error {{ambiguous cast from base 'A' to derived 'H':\n    struct A -> struct B -> struct G1 -> struct H\n    struct A -> struct B -> struct G2 -> struct H}}

  // TODO: Test DR427. This requires user-defined conversions, though.
}

// Enum conversions
void t_529_7()
{
  (void)(Enum)(1);
  (void)(Enum)(1.0);
  (void)(Onom)(En1);

  // Bad code below

  (void)(Enum)((int*)0); // expected-error {{C-style cast from 'int *' to 'Enum' is not allowed}}
}

// Void pointer to object pointer
void t_529_10()
{
  (void)(int*)((void*)0);
  (void)(const A*)((void*)0);
  (void)(int*)((const void*)0); // const_cast appended
}

// Member pointer upcast.
void t_529_9()
{
  (void)(int A::*)((int B::*)0);

  // Bad code below
  (void)(int A::*)((int H::*)0); // expected-error {{ambiguous conversion from pointer to member of derived class 'H' to pointer to member of base class 'A':}}
  (void)(int A::*)((int F::*)0); // expected-error {{conversion from pointer to member of class 'F' to pointer to member of class 'A' via virtual base 'B' is not allowed}}
}

// -------- reinterpret_cast -----------

enum test { testval = 1 };
struct structure { int m; };
typedef void (*fnptr)();

// Test conversion between pointer and integral types, as in p3 and p4.
void integral_conversion()
{
  void *vp = (void*)(testval);
  long l = (long)(vp);
  (void)(float*)(l);
  fnptr fnp = (fnptr)(l);
  (void)(char)(fnp); // expected-error {{cast from pointer to smaller type 'char' loses information}}
  (void)(long)(fnp);
}

void pointer_conversion()
{
  int *p1 = 0;
  float *p2 = (float*)(p1);
  structure *p3 = (structure*)(p2);
  typedef int **ppint;
  ppint *deep = (ppint*)(p3);
  (void)(fnptr*)(deep);
}

void constness()
{
  int ***const ipppc = 0;
  int const *icp = (int const*)(ipppc);
  (void)(int*)(icp); // const_cast appended
  int const *const **icpcpp = (int const* const**)(ipppc); // const_cast appended
  int *ip = (int*)(icpcpp);
  (void)(int const*)(ip);
  (void)(int const* const* const*)(ipppc);
}

void fnptrs()
{
  typedef int (*fnptr2)(int);
  fnptr fp = 0;
  (void)(fnptr2)(fp);
  void *vp = (void*)(fp);
  (void)(fnptr)(vp);
}

void refs()
{
  long l = 0;
  char &c = (char&)(l);
  // Bad: from rvalue
  (void)(int&)(&c); // expected-error {{C-style cast from rvalue to reference type 'int &'}}
}

void memptrs()
{
  const int structure::*psi = 0;
  (void)(const float structure::*)(psi);
  (void)(int structure::*)(psi); // const_cast appended

  void (structure::*psf)() = 0;
  (void)(int (structure::*)())(psf);

  (void)(void (structure::*)())(psi); // expected-error {{C-style cast from 'const int structure::*' to 'void (structure::*)()' is not allowed}}
  (void)(int structure::*)(psf); // expected-error {{C-style cast from 'void (structure::*)()' to 'int structure::*' is not allowed}}
}
