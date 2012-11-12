// RUN: %clang_cc1 -fsyntax-only -verify %s
// REQUIRES: LP64

// ------------ not interpreted as C-style cast ------------

struct SimpleValueInit {
  int i;
};

struct InitViaConstructor {
  InitViaConstructor(int i = 7);
};

struct NoValueInit { // expected-note 2 {{candidate constructor (the implicit copy constructor)}} 
  NoValueInit(int i, int j); // expected-note 2 {{candidate constructor}}
};

void test_cxx_functional_value_init() {
  (void)SimpleValueInit();
  (void)InitViaConstructor();
  (void)NoValueInit(); // expected-error{{no matching constructor for initialization}}
}

void test_cxx_function_cast_multi() { 
  (void)NoValueInit(0, 0);
  (void)NoValueInit(0, 0, 0); // expected-error{{no matching constructor for initialization}}
  (void)int(1, 2); // expected-error{{excess elements in scalar initializer}}
}


// ------------------ everything else --------------------

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
  char ***var2 = cppp(var);
  char ***const &var3 = var2;
  // Const reference to reference.
  char ***&var4 = cpppr(var3);
  // Drop reference. Intentionally without qualifier change.
  char *** var5 = cppp(var4);
  const int ar[100] = {0};
  // Array decay. Intentionally without qualifier change.
  typedef int *intp;
  int *pi = intp(ar);
  f fp = 0;
  // Don't misidentify fn** as a function pointer.
  typedef f *fp_t;
  f *fpp = fp_t(&fp);
  int const A::* const A::*icapcap = 0;
  typedef int A::* A::*iapap_t;
  iapap_t iapap = iapap_t(icapcap);
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
  (void)float(i);
  double d = 1.0;
  (void)float(d);
  (void)int(d);
  (void)char(i);
  typedef unsigned long ulong;
  (void)ulong(i);
  (void)int(En1);
  (void)double(En1);
  typedef int &intr;
  (void)intr(i);
  typedef const int &cintr;
  (void)cintr(i);

  int ar[1];
  typedef const int *cintp;
  (void)cintp(ar);
  typedef void (*pfvv)();
  (void)pfvv(t_529_2);

  typedef void *voidp;
  (void)voidp(0);
  (void)voidp((int*)0);
  typedef volatile const void *vcvoidp;
  (void)vcvoidp((const int*)0);
  typedef A *Ap;
  (void)Ap((B*)0);
  typedef A &Ar;
  (void)Ar(*((B*)0));
  typedef const B *cBp;
  (void)cBp((C1*)0);
  typedef B &Br;
  (void)Br(*((C1*)0));
  (void)Ap((D*)0);
  typedef const A &cAr;
  (void)cAr(*((D*)0));
  typedef int B::*Bmp;
  (void)Bmp((int A::*)0);
  typedef void (B::*Bmfp)();
  (void)Bmfp((void (A::*)())0);
  (void)Ap((E*)0); // functional-style cast ignores access control
  (void)voidp((const int*)0); // const_cast appended

  (void)int(Co1());
  (void)Co2(1);
  (void)Co3((Co4)(Co3()));

  // Bad code below
  //(void)(A*)((H*)0); // {{static_cast from 'struct H *' to 'struct A *' is not allowed}}
}

// Anything to void
void t_529_4()
{
  void(1);
  (void(t_529_4));
}

// Static downcasts
void t_529_5_8()
{
  typedef B *Bp;
  (void)Bp((A*)0);
  typedef B &Br;
  (void)Br(*((A*)0));
  typedef const G1 *cG1p;
  (void)cG1p((A*)0);
  typedef const G1 &cG1r;
  (void)cG1r(*((A*)0));
  (void)Bp((const A*)0); // const_cast appended
  (void)Br(*((const A*)0)); // const_cast appended
  typedef E *Ep;
  (void)Ep((A*)0); // access control ignored
  typedef E &Er;
  (void)Er(*((A*)0)); // access control ignored

  // Bad code below

  typedef C1 *C1p;
  (void)C1p((A*)0); // expected-error {{cannot cast 'A *' to 'C1p' (aka 'C1 *') via virtual base 'B'}}
  typedef C1 &C1r;
  (void)C1r(*((A*)0)); // expected-error {{cannot cast 'A' to 'C1r' (aka 'C1 &') via virtual base 'B'}}
  typedef D *Dp;
  (void)Dp((A*)0); // expected-error {{cannot cast 'A *' to 'Dp' (aka 'D *') via virtual base 'B'}}
  typedef D &Dr;
  (void)Dr(*((A*)0)); // expected-error {{cannot cast 'A' to 'Dr' (aka 'D &') via virtual base 'B'}}
  typedef H *Hp;
  (void)Hp((A*)0); // expected-error {{ambiguous cast from base 'A' to derived 'H':\n    struct A -> struct B -> struct G1 -> struct H\n    struct A -> struct B -> struct G2 -> struct H}}
  typedef H &Hr;
  (void)Hr(*((A*)0)); // expected-error {{ambiguous cast from base 'A' to derived 'H':\n    struct A -> struct B -> struct G1 -> struct H\n    struct A -> struct B -> struct G2 -> struct H}}

  // TODO: Test DR427. This requires user-defined conversions, though.
}

// Enum conversions
void t_529_7()
{
  (void)Enum(1);
  (void)Enum(1.0);
  (void)Onom(En1);

  // Bad code below

  (void)Enum((int*)0); // expected-error {{functional-style cast from 'int *' to 'Enum' is not allowed}}
}

// Void pointer to object pointer
void t_529_10()
{
  typedef int *intp;
  (void)intp((void*)0);
  typedef const A *cAp;
  (void)cAp((void*)0);
  (void)intp((const void*)0); // const_cast appended
}

// Member pointer upcast.
void t_529_9()
{
  typedef int A::*Amp;
  (void)Amp((int B::*)0);

  // Bad code below
  (void)Amp((int H::*)0); // expected-error {{ambiguous conversion from pointer to member of derived class 'H' to pointer to member of base class 'A':}}
  (void)Amp((int F::*)0); // expected-error {{conversion from pointer to member of class 'F' to pointer to member of class 'A' via virtual base 'B' is not allowed}}
}

// -------- reinterpret_cast -----------

enum test { testval = 1 };
struct structure { int m; };
typedef void (*fnptr)();

// Test conversion between pointer and integral types, as in p3 and p4.
void integral_conversion()
{
  typedef void *voidp;
  void *vp = voidp(testval);
  long l = long(vp);
  typedef float *floatp;
  (void)floatp(l);
  fnptr fnp = fnptr(l);
  (void)char(fnp); // expected-error {{cast from pointer to smaller type 'char' loses information}}
  (void)long(fnp);
}

void pointer_conversion()
{
  int *p1 = 0;
  typedef float *floatp;
  float *p2 = floatp(p1);
  typedef structure *structurep;
  structure *p3 = structurep(p2);
  typedef int **ppint;
  typedef ppint *pppint;
  ppint *deep = pppint(p3);
  typedef fnptr fnptrp;
  (void)fnptrp(deep);
}

void constness()
{
  int ***const ipppc = 0;
  typedef int const *icp_t;
  int const *icp = icp_t(ipppc);
  typedef int *intp;
  (void)intp(icp); // const_cast appended
  typedef int const *const ** intcpcpp;
  intcpcpp icpcpp = intcpcpp(ipppc); // const_cast appended
  int *ip = intp(icpcpp);
  (void)icp_t(ip);
  typedef int const *const *const *intcpcpcp;
  (void)intcpcpcp(ipppc);
}

void fnptrs()
{
  typedef int (*fnptr2)(int);
  fnptr fp = 0;
  (void)fnptr2(fp);
  typedef void *voidp;
  void *vp = voidp(fp);
  (void)fnptr(vp);
}

void refs()
{
  long l = 0;
  typedef char &charr;
  char &c = charr(l);
  // Bad: from rvalue
  typedef int &intr;
  (void)intr(&c); // expected-error {{functional-style cast from rvalue to reference type 'intr' (aka 'int &')}}
}

void memptrs()
{
  const int structure::*psi = 0;
  typedef const float structure::*structurecfmp;
  (void)structurecfmp(psi);
  typedef int structure::*structureimp;
  (void)structureimp(psi); // const_cast appended

  void (structure::*psf)() = 0;
  typedef int (structure::*structureimfp)();
  (void)structureimfp(psf);

  typedef void (structure::*structurevmfp)();
  (void)structurevmfp(psi); // expected-error {{functional-style cast from 'const int structure::*' to 'structurevmfp' (aka 'void (structure::*)()') is not allowed}}
  (void)structureimp(psf); // expected-error {{functional-style cast from 'void (structure::*)()' to 'structureimp' (aka 'int structure::*') is not allowed}}
}

// ---------------- misc ------------------

void crash_on_invalid_1()
{
  typedef itn Typo; // expected-error {{unknown type name 'itn'}}
  (void)Typo(1); // used to crash

  typedef int &int_ref;
  (void)int_ref(); // expected-error {{reference to type 'int' requires an initializer}}
}
