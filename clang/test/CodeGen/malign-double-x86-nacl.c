// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-nacl | FileCheck %s
// Check that i686-nacl essentially has -malign-double, which aligns
// double, long double, and long long to 64-bits.

int checksize[sizeof(long double) == 8 ? 1 : -1];
int checkalign[__alignof(long double) == 8 ? 1 : -1];

// CHECK-LABEL: define void @s1(double %a)
void s1(long double a) {}

struct st_ld {
  char c;
  long double ld;
};
int checksize2[sizeof(struct st_ld) == 16 ? 1 : -1];
int checkalign2[__alignof(struct st_ld) == 8 ? 1 : -1];

int checksize3[sizeof(double) == 8 ? 1 : -1];
int checkalign3[__alignof(double) == 8 ? 1 : -1];

// CHECK-LABEL: define void @s2(double %a)
void s2(double a) {}

struct st_d {
  char c;
  double d;
};
int checksize4[sizeof(struct st_d) == 16 ? 1 : -1];
int checkalign4[__alignof(struct st_d) == 8 ? 1 : -1];


int checksize5[sizeof(long long) == 8 ? 1 : -1];
int checkalign5[__alignof(long long) == 8 ? 1 : -1];

// CHECK-LABEL: define void @s3(i64 %a)
void s3(long long a) {}

struct st_ll {
  char c;
  long long ll;
};
int checksize6[sizeof(struct st_ll) == 16 ? 1 : -1];
int checkalign6[__alignof(struct st_ll) == 8 ? 1 : -1];
