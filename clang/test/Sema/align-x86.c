// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR3433
double g1;
short chk1[__alignof__(g1) == 8 ? 1 : -1]; 
short chk2[__alignof__(double) == 8 ? 1 : -1];

long long g2;
short chk1[__alignof__(g2) == 8 ? 1 : -1]; 
short chk2[__alignof__(long long) == 8 ? 1 : -1];

unsigned long long g5;
short chk1[__alignof__(g5) == 8 ? 1 : -1]; 
short chk2[__alignof__(unsigned long long) == 8 ? 1 : -1];

_Complex double g3;
short chk1[__alignof__(g3) == 8 ? 1 : -1]; 
short chk2[__alignof__(_Complex double) == 8 ? 1 : -1];

// PR6362
struct __attribute__((packed)) {unsigned int a;} g4;
short chk1[__alignof__(g4) == 1 ? 1 : -1];
short chk2[__alignof__(g4.a) == 1 ? 1 : -1];


// PR5637

#define ALIGNED(x) __attribute__((aligned(x)))

typedef ALIGNED(2) struct {
  char a[3];
} T;

short chk1[sizeof(T)       == 3 ? 1 : -1];
short chk2[sizeof(T[1])    == 4 ? 1 : -1];
short chk3[sizeof(T[2])    == 6 ? 1 : -1];
short chk4[sizeof(T[2][1]) == 8 ? 1 : -1];
short chk5[sizeof(T[1][2]) == 6 ? 1 : -1];

typedef struct ALIGNED(2) {
  char a[3];
} T2;

short chk1[sizeof(T2)       == 4 ? 1 : -1];
short chk2[sizeof(T2[1])    == 4 ? 1 : -1];
short chk3[sizeof(T2[2])    == 8 ? 1 : -1];
short chk4[sizeof(T2[2][1]) == 8 ? 1 : -1];
short chk5[sizeof(T2[1][2]) == 8 ? 1 : -1];
