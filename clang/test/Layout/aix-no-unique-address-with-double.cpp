// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only %s | \
// RUN:   FileCheck %s

struct Empty {};

struct A {
  double d;
};

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct Empty (empty)
// CHECK-NEXT:            | [sizeof=1, dsize=1, align=1, preferredalign=1,
// CHECK-NEXT:            |  nvsize=1, nvalign=1, preferrednvalign=1]

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct A
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

struct B {
  ~B();

  Empty emp;
  A a;
  char c;
};

struct B1 {
  [[no_unique_address]] B b;
  char ext[7];
};

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct B
// CHECK-NEXT:          0 |   struct Empty emp (empty)
// CHECK-NEXT:          4 |   struct A a
// CHECK-NEXT:          4 |     double d
// CHECK-NEXT:         12 |   char c
// CHECK-NEXT:            | [sizeof=16, dsize=13, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=13, nvalign=4, preferrednvalign=4]

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct B1
// CHECK-NEXT:          0 |   struct B b
// CHECK-NEXT:          0 |     struct Empty emp (empty)
// CHECK-NEXT:          4 |     struct A a
// CHECK-NEXT:          4 |       double d
// CHECK-NEXT:         12 |     char c
// CHECK-NEXT:         13 |   char [7] ext
// CHECK-NEXT:            | [sizeof=20, dsize=20, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=20, nvalign=4, preferrednvalign=4]

struct C {
  ~C();

  [[no_unique_address]] Empty emp;
  A a;
  char c;
};

struct C1 {
  [[no_unique_address]] C c;
  char ext[7];
};

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct C
// CHECK-NEXT:          0 |   struct Empty emp (empty)
// CHECK-NEXT:          0 |   struct A a
// CHECK-NEXT:          0 |     double d
// CHECK-NEXT:          8 |   char c
// CHECK-NEXT:            | [sizeof=16, dsize=9, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=9, nvalign=4, preferrednvalign=8]

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct C1
// CHECK-NEXT:          0 |   struct C c
// CHECK-NEXT:          0 |     struct Empty emp (empty)
// CHECK-NEXT:          0 |     struct A a
// CHECK-NEXT:          0 |       double d
// CHECK-NEXT:          8 |     char c
// CHECK-NEXT:          9 |   char [7] ext
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

struct D {
  ~D();

  [[no_unique_address]] char notEmp;
  A a;
  char c;
};

struct D1 {
  [[no_unique_address]] D d;
  char ext[7];
};

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct D
// CHECK-NEXT:          0 |   char notEmp
// CHECK-NEXT:          4 |   struct A a
// CHECK-NEXT:          4 |     double d
// CHECK-NEXT:         12 |   char c
// CHECK-NEXT:            | [sizeof=16, dsize=13, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=13, nvalign=4, preferrednvalign=4]

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct D1
// CHECK-NEXT:          0 |   struct D d
// CHECK-NEXT:          0 |     char notEmp
// CHECK-NEXT:          4 |     struct A a
// CHECK-NEXT:          4 |       double d
// CHECK-NEXT:         12 |     char c
// CHECK-NEXT:         13 |   char [7] ext
// CHECK-NEXT:            | [sizeof=20, dsize=20, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=20, nvalign=4, preferrednvalign=4]

struct E {
  [[no_unique_address]] Empty emp;
  int : 0;
  double d;
};

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct E
// CHECK-NEXT:          0 |   struct Empty emp (empty)
// CHECK-NEXT:        0:- |   int
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=4]

struct F {
  [[no_unique_address]] Empty emp, emp2;
  double d;
};

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT: 0 | struct F
// CHECK-NEXT: 0 |   struct Empty emp (empty)
// CHECK-NEXT: 1 |   struct Empty emp2 (empty)
// CHECK-NEXT: 0 |   double d
// CHECK-NEXT:   | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:   |  nvsize=8, nvalign=4, preferrednvalign=8]

int a = sizeof(Empty);
int b = sizeof(A);
int c = sizeof(B1);
int d = sizeof(C1);
int e = sizeof(D1);
int f = sizeof(E);
int g = sizeof(F);
