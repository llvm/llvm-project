// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s

typedef struct A {
	char x;
	int a : 22;
	int : 0;
	int c : 10;
	char b : 3;
	char d: 4;
	short y;
} A;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct A
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT: 4:0-21 |   int a
// CHECK-NEXT:    8:- |   int
// CHECK-NEXT:  8:0-9 |   int c
// CHECK-NEXT: 12:0-2 |   char b
// CHECK-NEXT: 12:3-6 |   char d
// CHECK-NEXT:     14 |   short y
// CHECK-NEXT:        | [sizeof=16, align=4]

typedef struct B {
	char x;
	int : 0;
	short a : 4;
	char y;
} B;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct B
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:    1:- |   int
// CHECK-NEXT:  2:0-3 |   short a
// CHECK-NEXT:      4 |   char y
// CHECK-NEXT:        | [sizeof=6, align=2]

typedef struct C {
	char x;
	short a : 4;
	int : 0;
	char y;
} C;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct C
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  2:0-3 |   short a
// CHECK-NEXT:    4:- |   int
// CHECK-NEXT:      4 |   char y
// CHECK-NEXT:        | [sizeof=8, align=4]

typedef struct D {
	char x;
	short : 0;
	int : 0;
	char y;
} D;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct D
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:    1:- |   short
// CHECK-NEXT:    1:- |   int
// CHECK-NEXT:      1 |   char y
// CHECK-NEXT:        | [sizeof=2, align=1]

typedef union E {
	char x;
	long long a : 3;
	int b : 3;
	long long : 0;
	short y;
} E;


// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | union E
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  0:0-2 |   long long a
// CHECK-NEXT:  0:0-2 |   int b
// CHECK-NEXT:    0:- |   long long
// CHECK-NEXT:      0 |   short
// CHECK-NEXT:        | [sizeof=8, align=2]

typedef struct F {
	char x;
	char a : 3;
	char b : 3;
	char c : 3;
	short d : 6;
	short e : 6;
	short f : 6;
	short g : 11;
	short h : 11;
	short i : 11;
	short y;
} F;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct F
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  1:0-2 |   char a
// CHECK-NEXT:  1:3-5 |   char b
// CHECK-NEXT:  2:0-2 |   char c
// CHECK-NEXT:  4:0-5 |   short d
// CHECK-NEXT: 4:6-11 |   short e
// CHECK-NEXT:  6:0-5 |   short f
// CHECK-NEXT: 8:0-10 |   short g
// CHECK-NEXT:10:0-10 |   short h
// CHECK-NEXT:12:0-10 |   short i
// CHECK-NEXT:     14 |   short y
// CHECK-NEXT:        | [sizeof=16, align=2]

typedef union G {
	char x;
	int a : 3;
	int : 0;
	long long : 0;
	short y;
} G;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | union G
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  0:0-2 |   int a
// CHECK-NEXT:    0:- |   int
// CHECK-NEXT:    0:- |   long long
// CHECK-NEXT:      0 |   short y
// CHECK-NEXT:        | [sizeof=4, align=2]

typedef struct H {
	unsigned short a : 1;
	unsigned char : 0;
	unsigned long : 0;
	unsigned short c : 1;
} H;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct H
// CHECK-NEXT:  0:0-0 |   unsigned short a
// CHECK-NEXT:    2:- |   unsigned char
// CHECK-NEXT:    2:- |   unsigned long
// CHECK-NEXT:  2:0-0 |   unsigned short c
// CHECK-NEXT:        | [sizeof=4, align=2]

typedef struct I {
	short : 8;
	__declspec(align(16)) short : 8;
} I;


// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct I
// CHECK-NEXT:  0:0-7 |   short
// CHECK-NEXT:  1:0-7 |   short
// CHECK-NEXT:        | [sizeof=2, align=2]

#pragma pack(push, 1)

typedef struct A1 {
	char x;
	int a : 22;
	int : 0;
	int c : 10;
	char b : 3;
	char d: 4;
	short y;
} A1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct A1
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT: 1:0-21 |   int a
// CHECK-NEXT:    5:- |   int
// CHECK-NEXT:  5:0-9 |   int c
// CHECK-NEXT:  9:0-2 |   char b
// CHECK-NEXT:  9:3-6 |   char d
// CHECK-NEXT:     10 |   short y
// CHECK-NEXT:        | [sizeof=12, align=1]

typedef struct B1 {
	char x;
	int : 0;
	short a : 4;
	char y;
} B1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct B1
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:    1:- |   int
// CHECK-NEXT:  1:0-3 |   short
// CHECK-NEXT:      3 |   char y
// CHECK-NEXT:        | [sizeof=4, align=1]

typedef struct C1 {
	char x;
	short a : 4;
	int : 0;
	char y;
} C1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct C1
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  1:0-3 |   short
// CHECK-NEXT:    3:- |   int
// CHECK-NEXT:      3 |   char y
// CHECK-NEXT:        | [sizeof=4, align=1]

typedef struct D1 {
	char x;
	short : 0;
	int : 0;
	char y;
} D1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct D1
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:    1:- |   short
// CHECK-NEXT:    1:- |   int
// CHECK-NEXT:      1 |   char y
// CHECK-NEXT:        | [sizeof=2, align=1]

typedef union E1 {
	char x;
	long long a : 3;
	int b : 3;
	long long : 0;
	short y;
} E1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | union E1
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  0:0-2 |   long long a
// CHECK-NEXT:  0:0-2 |   int b
// CHECK-NEXT:    0:- |   long long
// CHECK-NEXT:      0 |   short y
// CHECK-NEXT:        | [sizeof=8, align=1]

typedef struct F1 {
	char x;
	char a : 3;
	char b : 3;
	char c : 3;
	short d : 6;
	short e : 6;
	short f : 6;
	short g : 11;
	short h : 11;
	short i : 11;
	short y;
} F1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct F1
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  1:0-2 |   char a
// CHECK-NEXT:  1:3-5 |   char b
// CHECK-NEXT:  2:0-2 |   char c
// CHECK-NEXT:  3:0-5 |   short d
// CHECK-NEXT: 3:6-11 |   short e
// CHECK-NEXT:  5:0-5 |   short f
// CHECK-NEXT: 7:0-10 |   short g
// CHECK-NEXT: 9:0-10 |   short h
// CHECK-NEXT:11:0-10 |   short i
// CHECK-NEXT:     13 |   short y
// CHECK-NEXT:        | [sizeof=15, align=1]

typedef union G1 {
	char x;
	int a : 3;
	int : 0;
	long long : 0;
	short y;
} G1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | union G1
// CHECK-NEXT:      0 |   char x
// CHECK-NEXT:  0:0-2 |   int a
// CHECK-NEXT:    0:- |   int
// CHECK-NEXT:    0:- |   long long
// CHECK-NEXT:      0 |   short y
// CHECK-NEXT:        | [sizeof=4, align=1]

typedef struct H1 {
	unsigned long a : 1;
	unsigned char : 0;
	unsigned long : 0;
	unsigned long c : 1;
} H1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct H1
// CHECK-NEXT:  0:0-0 |   unsigned long a
// CHECK-NEXT:    4:- |   unsigned char
// CHECK-NEXT:    4:- |   unsigned long
// CHECK-NEXT:  4:0-0 |   unsigned long c
// CHECK-NEXT:        | [sizeof=8, align=1]

typedef struct I1 {
	short : 8;
	__declspec(align(16)) short : 8;
} I1;

// CHECK:*** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct I1
// CHECK-NEXT:  0:0-7 |   short
// CHECK-NEXT:  1:0-7 |   short
// CHECK-NEXT:        | [sizeof=2, align=1]

#pragma pack(pop)

int x[
sizeof(A ) +
sizeof(B ) +
sizeof(C ) +
sizeof(D ) +
sizeof(E ) +
sizeof(F ) +
sizeof(G ) +
sizeof(H ) +
sizeof(I ) +
sizeof(A1) +
sizeof(B1) +
sizeof(C1) +
sizeof(D1) +
sizeof(E1) +
sizeof(F1) +
sizeof(G1) +
sizeof(H1) +
sizeof(I1) +
0];
