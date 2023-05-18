// Test this without pch.
// RUN: %clang_cc1 -include %s -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

#ifndef HEADER
#define HEADER

extern float y;
extern int *ip, x;

float z;

int z2 = 17;

#define MAKE_HAPPY(X) X##Happy
int MAKE_HAPPY(Very);

#define A_MACRO_IN_THE_PCH 492
#define FUNCLIKE_MACRO(X, Y) X ## Y

#define PASTE2(x,y) x##y
#define PASTE1(x,y) PASTE2(x,y)
#define UNIQUE(x) PASTE1(x,__COUNTER__)

int UNIQUE(a);  // a0
int UNIQUE(a);  // a1

#else

int *ip2 = &x;
float *fp = &ip; // expected-warning{{incompatible pointer types}}
double z; // expected-error{{redefinition}} expected-note@14{{previous}}
int z2 = 18; // expected-error{{redefinition}} expected-note@16{{previous}}
double VeryHappy; // expected-error{{redefinition}} expected-note@19{{previous definition is here}}

int Q = A_MACRO_IN_THE_PCH;

int R = FUNCLIKE_MACRO(A_MACRO_, IN_THE_PCH);


int UNIQUE(a);  // a2
int *Arr[] = { &a0, &a1, &a2 };

#endif
