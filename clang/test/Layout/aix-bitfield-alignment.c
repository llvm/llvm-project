// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s

struct A {
  unsigned char c : 2;
} A;

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct A
// CHECK-NEXT:      0:0-1 |   unsigned char c
// CHECK-NEXT:               sizeof=4, {{(dsize=4, )?}}align=4, preferredalign=4

struct B {
  char c;
  int : 0;
} B;

int b = sizeof(B);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct B
// CHECK-NEXT:          0 |   char c
// CHECK-NEXT:        4:- |   int
// CHECK-NEXT:               sizeof=4, {{(dsize=4, )?}}align=4, preferredalign=4

struct C {
  signed int a1 : 6;
  signed char a2 : 4;
  short int a3 : 2;
  int a4 : 2;
  signed long a5 : 5;
  long long int a6 : 6;
  unsigned long a7 : 8;
} C;

int c = sizeof(C);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct C
// CHECK-NEXT:      0:0-5 |   int a1
// CHECK-NEXT:      0:6-9 |   signed char a2
// CHECK-NEXT:      1:2-3 |   short a3
// CHECK-NEXT:      1:4-5 |   int a4
// CHECK-NEXT:     1:6-10 |   long a5
// CHECK-NEXT:      2:3-8 |   long long a6
// CHECK32:         4:0-7 |   unsigned long a7
// CHECK32:                  sizeof=8, {{(dsize=8, )?}}align=4, preferredalign=4
// CHECK64:         3:1-8 |   unsigned long a7
// CHECK64:                  sizeof=8, {{(dsize=8, )?}}align=8, preferredalign=8

#pragma align(packed)
struct C1 {
  signed int a1 : 6;
  signed char a2 : 4;
  short int a3 : 2;
  int a4 : 2;
  signed long a5 : 5;
  long long int a6 : 6;
  unsigned long a7 : 8;
} C1;
#pragma align(reset)

int c1 = sizeof(C1);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct C1
// CHECK-NEXT:      0:0-5 |   int a1
// CHECK-NEXT:      0:6-9 |   signed char a2
// CHECK-NEXT:      1:2-3 |   short a3
// CHECK-NEXT:      1:4-5 |   int a4
// CHECK-NEXT:     1:6-10 |   long a5
// CHECK-NEXT:      2:3-8 |   long long a6
// CHECK-NEXT:      3:1-8 |   unsigned long a7
// CHECK-NEXT:               sizeof=5, {{(dsize=5, )?}}align=1, preferredalign=1

#pragma pack(4)
struct C2 {
  signed int a1 : 6;
  signed char a2 : 4;
  short int a3 : 2;
  int a4 : 2;
  signed long a5 : 5;
  long long int a6 : 6;
  unsigned long a7 : 8;
} C2;
#pragma pack(pop)

int c2 = sizeof(C2);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct C2
// CHECK-NEXT:      0:0-5 |   int a1
// CHECK-NEXT:      0:6-9 |   signed char a2
// CHECK-NEXT:      1:2-3 |   short a3
// CHECK-NEXT:      1:4-5 |   int a4
// CHECK-NEXT:     1:6-10 |   long a5
// CHECK-NEXT:      2:3-8 |   long long a6
// CHECK-NEXT:      3:1-8 |   unsigned long a7
// CHECK-NEXT:               sizeof=8, {{(dsize=8, )?}}align=4, preferredalign=4

typedef __attribute__((aligned(32))) short mySHORT;
struct D {
  char c : 8;
  mySHORT : 0;
} D;

int d = sizeof(D);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct D
// CHECK-NEXT:      0:0-7 |   char c
// CHECK-NEXT:       32:- |   mySHORT
// CHECK-NEXT:               sizeof=32, {{(dsize=32, )?}}align=32, preferredalign=32

typedef __attribute__((aligned(32))) long myLONG;
struct D11 {
  char c : 8;
  myLONG : 0;
} D11;

int d11 = sizeof(D11);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct D11
// CHECK-NEXT:      0:0-7 |   char c
// CHECK-NEXT:       32:- |   myLONG
// CHECK-NEXT:               sizeof=32, {{(dsize=32, )?}}align=32, preferredalign=32

typedef __attribute__((aligned(2))) long myLONG2;
struct D12 {
  char c : 8;
  myLONG2 : 0;
} D12;

int d12 = sizeof(D12);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct D12
// CHECK-NEXT:      0:0-7 |   char c
// CHECK32:           4:- |   myLONG2
// CHECK32:                  sizeof=4, {{(dsize=4, )?}}align=4, preferredalign=4
// CHECK64:           8:- |   myLONG2
// CHECK64:                  sizeof=8, {{(dsize=8, )?}}align=8, preferredalign=8

typedef __attribute__((aligned(32))) long long myLONGLONG;
struct D21 {
  char c : 8;
  myLONGLONG : 0;
} D21;

int d21 = sizeof(D21);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct D21
// CHECK-NEXT:      0:0-7 |   char c
// CHECK-NEXT:       32:- |   myLONGLONG
// CHECK-NEXT:               sizeof=32, {{(dsize=32, )?}}align=32, preferredalign=32

typedef __attribute__((aligned(2))) long long myLONGLONG2;
struct D22 {
  char c : 8;
  myLONGLONG2 : 0;
} D22;

int d22 = sizeof(D22);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct D22
// CHECK-NEXT:      0:0-7 |   char c
// CHECK32:           4:- |   myLONGLONG2
// CHECK32:                  sizeof=4, {{(dsize=4, )?}}align=4, preferredalign=4
// CHECK64:           8:- |   myLONGLONG2
// CHECK64:                  sizeof=8, {{(dsize=8, )?}}align=8, preferredalign=8

enum LL : unsigned long long { val = 1 };

struct E {
  enum LL e : 32;
} E;

int e = sizeof(E);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct E
// CHECK-NEXT:     0:0-31 |   enum LL e
// CHECK32-NEXT:             sizeof=4, {{(dsize=4, )?}}align=4, preferredalign=4
// CHECK64-NEXT:             sizeof=8, {{(dsize=8, )?}}align=8, preferredalign=8

enum LL1 : unsigned long long { val1 = 1 } __attribute__((aligned(16)));
struct E1 {
  enum LL1 e : 32;
} E1;

int e1 = sizeof(E1);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct E1
// CHECK-NEXT:     0:0-31 |   enum LL1 e
// CHECK-NEXT:               sizeof=16, {{(dsize=16, )?}}align=16, preferredalign=16

struct F {
  long long l : 32 __attribute__((aligned(16)));
} F;

int f = sizeof(F);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct F
// CHECK-NEXT:     0:0-31 |   long long l
// CHECK-NEXT:               sizeof=16, {{(dsize=16, )?}}align=16, preferredalign=16

struct G {
  long long ll : 45;
} G;

int s = sizeof(G);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct G
// CHECK-NEXT:     0:0-44 |   long long ll
// CHECK-NEXT:               sizeof=8, {{(dsize=8, )?}}align=8, preferredalign=8

#pragma align(packed)
struct H {
   char c;
   int : 0;
   int i;
} H;
#pragma align(reset)

int h = sizeof(H);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct H
// CHECK-NEXT:          0 |   char c
// CHECK-NEXT:        4:- |   int
// CHECK-NEXT:          4 |   int i
// CHECK-NEXT:              sizeof=8, {{(dsize=8, )?}}align=1, preferredalign=1

#pragma pack(2)
struct I {
   char c;
   int : 0;
   int i;
} I;
#pragma pack(pop)

int i = sizeof(I);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct I
// CHECK-NEXT:          0 |   char c
// CHECK-NEXT:        4:- |   int
// CHECK-NEXT:          4 |   int i
// CHECK-NEXT:              sizeof=8, {{(dsize=8, )?}}align=2, preferredalign=2
