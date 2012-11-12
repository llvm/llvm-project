// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -fsyntax-only -verify -ffreestanding
// expected-no-diagnostics
// <rdar://problem/10494810> and PR9560
// Check #pragma pack handling with bitfields.

#include <stddef.h>
#pragma pack(2)

struct s0 {
     char        f1;
     unsigned    f2 : 32;
     char        f3;
};
extern int check[sizeof(struct s0) == 6 ? 1 : -1];

struct s1 {
     char        f1;
     unsigned       : 0;
     char        f3;
};
extern int check[sizeof(struct s1) == 5 ? 1 : -1];

struct s2 {
     char        f1;
     unsigned       : 0;
     unsigned    f3 : 8;
     char        f4;
};
extern int check[sizeof(struct s2) == 6 ? 1 : -1];

struct s3 {
     char        f1;
     unsigned       : 0;
     unsigned    f3 : 16;
     char        f4;
};
extern int check[sizeof(struct s3) == 8 ? 1 : -1];
extern int check[offsetof(struct s3, f4) == 6 ? 1 : -1];

struct s4 {
     char        f1;
     unsigned    f2 : 8;
     char        f3;
};
extern int check[sizeof(struct s4) == 4 ? 1 : -1];
extern int check[offsetof(struct s4, f3) == 2 ? 1 : -1];
