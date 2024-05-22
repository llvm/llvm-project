// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -target-abi apcs-gnu -triple armv7-apple-darwin10 %s -verify
// expected-no-diagnostics
//
// Note: gcc forces the alignment to 4 bytes, regardless of the type of the
// zero length bitfield.

#include <stddef.h>

struct t1
{
  int foo : 1;
  char : 0;
  char bar;
};
static int arr1_offset[(offsetof(struct t1, bar) == 4) ? 0 : -1];
static int arr1_sizeof[(sizeof(struct t1) == 8) ? 0 : -1];

struct t2
{
  int foo : 1;
  short : 0;
  char bar;
};
static int arr2_offset[(offsetof(struct t2, bar) == 4) ? 0 : -1];
static int arr2_sizeof[(sizeof(struct t2) == 8) ? 0 : -1];

struct t3
{
  int foo : 1;
  int : 0;
  char bar;
};
static int arr3_offset[(offsetof(struct t3, bar) == 4) ? 0 : -1];
static int arr3_sizeof[(sizeof(struct t3) == 8) ? 0 : -1];

struct t4
{
  int foo : 1;
  long : 0;
  char bar;
};
static int arr4_offset[(offsetof(struct t4, bar) == 4) ? 0 : -1];
static int arr4_sizeof[(sizeof(struct t4) == 8) ? 0 : -1];

struct t5
{
  int foo : 1;
  long long : 0;
  char bar;
};
static int arr5_offset[(offsetof(struct t5, bar) == 4) ? 0 : -1];
static int arr5_sizeof[(sizeof(struct t5) == 8) ? 0 : -1];

struct t6
{
  int foo : 1;
  char : 0;
  char bar : 1;
  char bar2;
};
static int arr6_offset[(offsetof(struct t6, bar2) == 5) ? 0 : -1];
static int arr6_sizeof[(sizeof(struct t6) == 8) ? 0 : -1];

struct t7
{
  int foo : 1;
  short : 0;
  char bar1 : 1;
  char bar2;
};
static int arr7_offset[(offsetof(struct t7, bar2) == 5) ? 0 : -1];
static int arr7_sizeof[(sizeof(struct t7) == 8) ? 0 : -1];

struct t8
{
  int foo : 1;
  int : 0;
  char bar1 : 1;
  char bar2;
};
static int arr8_offset[(offsetof(struct t8, bar2) == 5) ? 0 : -1];
static int arr8_sizeof[(sizeof(struct t8) == 8) ? 0 : -1];

struct t9
{
  int foo : 1;
  long : 0;
  char bar1 : 1;
  char bar2;
};
static int arr9_offset[(offsetof(struct t9, bar2) == 5) ? 0 : -1];
static int arr9_sizeof[(sizeof(struct t9) == 8) ? 0 : -1];

struct t10
{
  int foo : 1;
  long long : 0;
  char bar1 : 1;
  char bar2;
};
static int arr10_offset[(offsetof(struct t10, bar2) == 5) ? 0 : -1];
static int arr10_sizeof[(sizeof(struct t10) == 8) ? 0 : -1];

struct t11
{
  int foo : 1;
  long long : 0;
  char : 0;
  char bar1 : 1;
  char bar2;
};
static int arr11_offset[(offsetof(struct t11, bar2) == 5) ? 0 : -1];
static int arr11_sizeof[(sizeof(struct t11) == 8) ? 0 : -1];

struct t12
{
  int foo : 1;
  char : 0;
  long long : 0;
  char : 0;
  char bar;
};
static int arr12_offset[(offsetof(struct t12, bar) == 4) ? 0 : -1];
static int arr12_sizeof[(sizeof(struct t12) == 8) ? 0 : -1];

struct t13
{
  char foo;
  long : 0;
  char bar;
};
static int arr13_offset[(offsetof(struct t13, bar) == 4) ? 0 : -1];
static int arr13_sizeof[(sizeof(struct t13) == 8) ? 0 : -1];

struct t14
{
  char foo1;
  int : 0;
  char foo2 : 1;
  short foo3 : 16;
  char : 0;
  short foo4 : 16;
  char bar1;
  int : 0;
  char bar2;
};
static int arr14_bar1_offset[(offsetof(struct t14, bar1) == 10) ? 0 : -1];
static int arr14_bar2_offset[(offsetof(struct t14, bar2) == 12) ? 0 : -1];
static int arr14_sizeof[(sizeof(struct t14) == 16) ? 0 : -1];

struct t15
{
  char foo;
  char : 0;
  int : 0;
  char bar;
  long : 0;
  char : 0;
};
static int arr15_offset[(offsetof(struct t15, bar) == 4) ? 0 : -1];
static int arr15_sizeof[(sizeof(struct t15) == 8) ? 0 : -1];

struct t16
{
  long : 0;
  char bar;
};
static int arr16_offset[(offsetof(struct t16, bar) == 0) ? 0 : -1];
static int arr16_sizeof[(sizeof(struct t16) == 4) ? 0 : -1];

struct t17
{
  char foo;
  long : 0;
  long : 0;
  char : 0;
  char bar;
};
static int arr17_offset[(offsetof(struct t17, bar) == 4) ? 0 : -1];
static int arr17_sizeof[(sizeof(struct t17) == 8) ? 0 : -1];

struct t18
{
  long : 0;
  long : 0;
  char : 0;
};
static int arr18_sizeof[(sizeof(struct t18) == 0) ? 0 : -1];

struct t19
{
  char foo1;
  long foo2 : 1;
  char : 0;
  long foo3 : 32;
  char bar;
};
static int arr19_offset[(offsetof(struct t19, bar) == 8) ? 0 : -1];
static int arr19_sizeof[(sizeof(struct t19) == 12) ? 0 : -1];

struct t20
{
  short : 0;
  int foo : 1;
  long : 0;
  char bar;
};
static int arr20_offset[(offsetof(struct t20, bar) == 4) ? 0 : -1];
static int arr20_sizeof[(sizeof(struct t20) == 8) ? 0 : -1];

struct t21
{
  short : 0;
  int foo1 : 1;
  char : 0;
  int foo2 : 16;
  long : 0;
  char bar1;
  int bar2;
  long bar3;
  char foo3 : 8;
  char : 0;
  long : 0;
  int foo4 : 32;
  short foo5: 1;
  long bar4;
  short foo6: 16;
  short foo7: 16;
  short foo8: 16;
};
static int arr21_bar1_offset[(offsetof(struct t21, bar1) == 8) ? 0 : -1];
static int arr21_bar2_offset[(offsetof(struct t21, bar2) == 12) ? 0 : -1];
static int arr21_bar3_offset[(offsetof(struct t21, bar3) == 16) ? 0 : -1];
static int arr21_bar4_offset[(offsetof(struct t21, bar4) == 32) ? 0 : -1];
static int arr21_sizeof[(sizeof(struct t21) == 44) ? 0 : -1];

int main() {
  return 0;
}

