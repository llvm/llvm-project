// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// Assignment of function pointers.

struct A
{
};

struct B1 : A
{
};

struct B2 : A
{
};

struct D : B1, B2
{
};

struct P : private A
{
};

// Some functions to play with below.
void s1() throw();
void s2() throw(int);
void s3() throw(A);
void s4() throw(B1);
void s5() throw(D);
void s6();
void s7() throw(int, float);
void (*s8())() throw(B1); // s8 returns a pointer to function with spec
void s9(void (*)() throw(B1)); // s9 takes pointer to function with spec

void s10() noexcept;
void s11() noexcept(true);
void s12() noexcept(false);

void fnptrs()
{
  // Assignment and initialization of function pointers.
  void (*t1)() throw() = &s1;    // valid
  t1 = &s2;                      // expected-error {{not superset}} expected-error {{incompatible type}}
  t1 = &s3;                      // expected-error {{not superset}} expected-error {{incompatible type}}
  void (&t2)() throw() = s2;     // expected-error {{not superset}}
  void (*t3)() throw(int) = &s2; // valid
  void (*t4)() throw(A) = &s1;   // valid
  t4 = &s3;                      // valid
  t4 = &s4;                      // valid
  t4 = &s5;                      // expected-error {{not superset}} expected-error {{incompatible type}}
  void (*t5)() = &s1;            // valid
  t5 = &s2;                      // valid
  t5 = &s6;                      // valid
  t5 = &s7;                      // valid
  t1 = t3;                       // expected-error {{not superset}} expected-error {{incompatible type}}
  t3 = t1;                       // valid
  void (*t6)() throw(B1);
  t6 = t4;                       // expected-error {{not superset}} expected-error {{incompatible type}}
  t4 = t6;                       // valid
  t5 = t1;                       // valid
  t1 = t5;                       // expected-error {{not superset}} expected-error {{incompatible type}}

  // return types and arguments must match exactly, no inheritance allowed
  void (*(*t7)())() throw(B1) = &s8;       // valid
  void (*(*t8)())() throw(A) = &s8;        // expected-error {{return types differ}}
  void (*(*t9)())() throw(D) = &s8;        // expected-error {{return types differ}}
  void (*t10)(void (*)() throw(B1)) = &s9; // valid
  void (*t11)(void (*)() throw(A)) = &s9;  // expected-error {{argument types differ}}
  void (*t12)(void (*)() throw(D)) = &s9;  // expected-error {{argument types differ}}
}

// Member function stuff

struct Str1 { void f() throw(int); }; // expected-note {{previous declaration}}
void Str1::f() // expected-warning {{missing exception specification}}
{
}

void mfnptr()
{
  void (Str1::*pfn1)() throw(int) = &Str1::f; // valid
  void (Str1::*pfn2)() = &Str1::f; // valid
  void (Str1::*pfn3)() throw() = &Str1::f; // expected-error {{not superset}}
}
