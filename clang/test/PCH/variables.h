// RUN: %clang_cc1 -emit-pch -o variables.h.pch variables.h
// Do not mess with the whitespace in this file. It's important.




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
