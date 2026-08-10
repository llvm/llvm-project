// RUN: %clang_cc1 -emit-llvm %s -o -
// PR2987
struct S2045
{
  unsigned short int a;
  union { } b;
  union __attribute__ ((aligned (4))) { } c[0];
};
struct S2045 s2045;
