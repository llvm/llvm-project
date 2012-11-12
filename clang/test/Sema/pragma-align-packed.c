// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s
// expected-no-diagnostics

#pragma pack(push, 1)
struct s0 {
  char f0;
  int  f1 __attribute__((aligned(4)));
};
extern int a[sizeof(struct s0) == 5 ? 1 : -1];
#pragma pack(pop)

struct __attribute__((packed)) s1 {
  char f0;
  int  f1 __attribute__((aligned(4)));
};
extern int a[sizeof(struct s1) == 8 ? 1 : -1];

#pragma options align=packed
struct s2 {
  char f0;
  int  f1 __attribute__((aligned(4)));
};
extern int a[sizeof(struct s2) == 5 ? 1 : -1];
#pragma options align=reset

#pragma pack(1)
struct s3_0 { unsigned char f0; unsigned int f1; };
int t3_0[sizeof(struct s3_0) == 5 ? 1 : -1];
#pragma options align=reset
struct s3_1 { unsigned char f0; unsigned int f1; };
int t3_1[sizeof(struct s3_1) == 8 ? 1 : -1];
