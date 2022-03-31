// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -Wno-strict-prototypes -verify %s

// PR3588
void g0(int, int);
void g0(); // expected-note{{previous declaration is here}} expected-note{{'g0' declared here}}

void f0(void) {
  g0(1, 2, 3); // expected-error{{too many arguments to function call}}
}

void g0(int); // expected-error{{conflicting types for 'g0'}}

int g1(int, int);

typedef int INT;

INT g1(x, y)
     int x;
     int y;
{
  return x + y;
}

int g2(int, int); // expected-note{{previous declaration is here}}

INT g2(x) // expected-error{{conflicting types for 'g2'}}
     int x;
{
  return x;
}

void test(void) {
  int f1;
  {
    void f1(double);
    {
      void f1(double); // expected-note{{previous declaration is here}}
      {
        int f1(int); // expected-error{{conflicting types for 'f1'}}
      }
    }
  }
}

extern void g3(int); // expected-note{{previous declaration is here}}
static void g3(int x) { } // expected-error{{static declaration of 'g3' follows non-static declaration}}

void test2(void) {
  extern int f2; // expected-note 2 {{previous definition is here}}
  {
    void f2(int); // expected-error{{redefinition of 'f2' as different kind of symbol}}
  }

  {
    int f2;
    {
      void f2(int); // expected-error{{redefinition of 'f2' as different kind of symbol}}
    }
  }
}

// <rdar://problem/6127293>
int outer1(int); // expected-note{{previous declaration is here}}
struct outer3 { };
int outer4(int); // expected-note{{previous declaration is here}}
int outer5; // expected-note{{previous definition is here}}
int *outer7(int);

void outer_test(void) {
  int outer1(float); // expected-error{{conflicting types for 'outer1'}}
  int outer2(int); // expected-note{{previous declaration is here}}
  int outer3(int); // expected-note{{previous declaration is here}}
  int outer4(int);
  int outer5(int); // expected-error{{redefinition of 'outer5' as different kind of symbol}}
  int* outer6(int); // expected-note{{previous declaration is here}}
  int *outer7(int);
  int outer8(int);

  int *ip7 = outer7(6);
}

int outer2(float); // expected-error{{conflicting types for 'outer2'}}
int outer3(float); // expected-error{{conflicting types for 'outer3'}}
int outer4(float); // expected-error{{conflicting types for 'outer4'}}

void outer_test2(int x) {
  int* ip = outer6(x); // expected-warning{{use of out-of-scope declaration of 'outer6'}}
  int *ip2 = outer7(x);
}

void outer_test3(void) {
  int *(*fp)(int) = outer8; // expected-error{{use of undeclared identifier 'outer8'}}
}

enum e { e1, e2 };

// GNU extension: prototypes and K&R function definitions
int isroot(short x, // expected-note{{previous declaration is here}}
           enum e); 

int isroot(x, y)
     short x; // expected-warning{{promoted type 'int' of K&R function parameter is not compatible with the parameter type 'short' declared in a previous prototype}}
     unsigned int y;
{
  return x == 1;
}

// PR3817
void *h0(unsigned a0,     ...);
extern __typeof (h0) h1 __attribute__((__sentinel__));
extern __typeof (h1) h1 __attribute__((__sentinel__));

// PR3840
void i0 (unsigned short a0);
extern __typeof (i0) i1;
extern __typeof (i1) i1;

typedef int a();
typedef int a2(int*);
a x;
a2 x2; // expected-note{{passing argument to parameter here}}
void test_x(void) {
  x(5);
  x2(5); // expected-warning{{incompatible integer to pointer conversion passing 'int' to parameter of type 'int *'}}
}

enum e0 {one}; 
void f3(); 
void f3(enum e0 x) {}

enum incomplete_enum;
void f4(); // expected-note {{previous declaration is here}}
void f4(enum incomplete_enum); // expected-error {{conflicting types for 'f4'}}
