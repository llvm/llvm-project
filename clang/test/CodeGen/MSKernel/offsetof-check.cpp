// Check that MY_OFFSETOF can be statically evaluated and
// returns the same results as __builtin_offsetof
// RUN: %clang_cc1 -fms-kernel  -fms-extensions -O2 -triple x86_64-windows-msvc -fsyntax-only %s -o /dev/null

typedef unsigned long long ULONG_PTR;

#define OFFSETOF(type, field) (ULONG_PTR)(&((type *)0)->field)
#define OFFSET_CHECK(type, field) \
	static_assert(OFFSETOF(type, field) == __builtin_offsetof(type, field))

typedef struct MyStruct1 {
  char b;
} MyStruct1;

typedef struct MyStruct2 {
  char b;
  long c;
} MyStruct2;

typedef struct MyStruct3 {
  char b;
  long x[10];
} MyStruct3;

typedef struct MyStruct4 {
  char c[10][10];
  alignas(16) int v;
} MyStruct4;

typedef struct MyStruct5 {
  char b;
  struct X {
   long m0[10], m1;
  } y;
  long c;
} MyStruct5;

typedef struct MyStruct6 {
  char b;
  struct X {
   long m0[10], m1;
  } x[10];
  long c;
} MyStruct6;

OFFSET_CHECK(MyStruct1, b);
OFFSET_CHECK(MyStruct2, b);
OFFSET_CHECK(MyStruct2, c);
OFFSET_CHECK(MyStruct3, b);
OFFSET_CHECK(MyStruct3, x[1]);
OFFSET_CHECK(MyStruct3, x[2]);
OFFSET_CHECK(MyStruct3, x[3]);
OFFSET_CHECK(MyStruct4, c[1][1]);
OFFSET_CHECK(MyStruct4, v);
OFFSET_CHECK(MyStruct5, b);
OFFSET_CHECK(MyStruct5, y);
OFFSET_CHECK(MyStruct5, y.m0[3]);
OFFSET_CHECK(MyStruct5, y.m1);
OFFSET_CHECK(MyStruct5, c);
OFFSET_CHECK(MyStruct6, b);
OFFSET_CHECK(MyStruct6, x);
OFFSET_CHECK(MyStruct6, x[1]);
OFFSET_CHECK(MyStruct6, x[3].m0[3]);
OFFSET_CHECK(MyStruct6, x[4].m1);
OFFSET_CHECK(MyStruct6, c);

