// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s
// expected-no-diagnostics

#include <stddef.h>

#pragma options align=mac68k

typedef float __attribute__((vector_size (8))) v2f_t;
typedef float __attribute__((vector_size (16))) v4f_t;

extern int a0_0[__alignof(v2f_t) == 8 ? 1 : -1];
extern int a0_1[__alignof(v4f_t) == 16 ? 1 : -1];

struct s1 {
  char f0;
  int  f1;
};
extern int a1_0[offsetof(struct s1, f0) == 0 ? 1 : -1];
extern int a1_1[offsetof(struct s1, f1) == 2 ? 1 : -1];
extern int a1_2[sizeof(struct s1) == 6 ? 1 : -1];
extern int a1_3[__alignof(struct s1) == 2 ? 1 : -1];

struct s2 {
  char f0;
  double f1;
};
extern int a2_0[offsetof(struct s2, f0) == 0 ? 1 : -1];
extern int a2_1[offsetof(struct s2, f1) == 2 ? 1 : -1];
extern int a2_2[sizeof(struct s2) == 10 ? 1 : -1];
extern int a2_3[__alignof(struct s2) == 2 ? 1 : -1];

struct s3 {
  char f0;
  v4f_t f1;
};
extern int a3_0[offsetof(struct s3, f0) == 0 ? 1 : -1];
extern int a3_1[offsetof(struct s3, f1) == 2 ? 1 : -1];
extern int a3_2[sizeof(struct s3) == 18 ? 1 : -1];
extern int a3_3[__alignof(struct s3) == 2 ? 1 : -1];

struct s4 {
  char f0;
  char f1;
};
extern int a4_0[offsetof(struct s4, f0) == 0 ? 1 : -1];
extern int a4_1[offsetof(struct s4, f1) == 1 ? 1 : -1];
extern int a4_2[sizeof(struct s4) == 2 ? 1 : -1];
extern int a4_3[__alignof(struct s4) == 2 ? 1 : -1];

struct s5 {
  unsigned f0 : 9;
  unsigned f1 : 9;
};
extern int a5_0[sizeof(struct s5) == 4 ? 1 : -1];
extern int a5_1[__alignof(struct s5) == 2 ? 1 : -1];

struct s6 {
  unsigned : 0;
  unsigned : 0;
};
extern int a6_0[sizeof(struct s6) == 0 ? 1 : -1];
extern int a6_1[__alignof(struct s6) == 2 ? 1 : -1];

struct s7 {
  char : 1;
  unsigned : 1;
};
extern int a7_0[sizeof(struct s7) == 2 ? 1 : -1];
extern int a7_1[__alignof(struct s7) == 2 ? 1 : -1];

struct s8 {
  char f0;
  unsigned : 1;
};
extern int a8_0[sizeof(struct s8) == 2 ? 1 : -1];
extern int a8_1[__alignof(struct s8) == 2 ? 1 : -1];

struct s9 {
  char f0[3];
  unsigned : 0;
  char f1;
};
extern int a9_0[sizeof(struct s9) == 6 ? 1 : -1];
extern int a9_1[__alignof(struct s9) == 2 ? 1 : -1];

struct s10 {
  char f0;
};
extern int a10_0[sizeof(struct s10) == 2 ? 1 : -1];
extern int a10_1[__alignof(struct s10) == 2 ? 1 : -1];

struct s11 {
  char f0;
  v2f_t f1;
};
extern int a11_0[offsetof(struct s11, f0) == 0 ? 1 : -1];
extern int a11_1[offsetof(struct s11, f1) == 2 ? 1 : -1];
extern int a11_2[sizeof(struct s11) == 10 ? 1 : -1];
extern int a11_3[__alignof(struct s11) == 2 ? 1 : -1];

#pragma options align=reset

void f12(void) {
  #pragma options align=mac68k
  struct s12 {
    char f0;
    int  f1;
  };
  #pragma options align=reset
  extern int a12[sizeof(struct s12) == 6 ? 1 : -1];
}
