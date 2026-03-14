// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fxl-pragma-pack -fsyntax-only %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fxl-pragma-pack -fsyntax-only %s | \
// RUN:   FileCheck %s

namespace test1 {
#pragma align(natural)
struct A {
  int i1;
};

struct B {
  double d1;
};
#pragma align(reset)

struct C : A, B {
  double d2;
};

int a = sizeof(C);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::A
// CHECK-NEXT:          0 |   int i1
// CHECK-NEXT:            | [sizeof=4, dsize=4, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4, preferrednvalign=4]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::B
// CHECK-NEXT:          0 |   double d1
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::C
// CHECK-NEXT:          0 |   struct test1::A (base)
// CHECK-NEXT:          0 |     int i1
// CHECK-NEXT:          4 |   struct test1::B (base)
// CHECK-NEXT:          4 |     double d1
// CHECK-NEXT:         12 |   double d2
// CHECK-NEXT:            | [sizeof=20, dsize=20, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=20, nvalign=4, preferrednvalign=4]

} // namespace test1

namespace test2 {
struct A {
  int i1;
  double d;
};

#pragma align(natural)
struct B : A {
  int i2;
};
#pragma align(reset)

int b = sizeof(B);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::A
// CHECK-NEXT:          0 |   int i1
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::B
// CHECK-NEXT:          0 |   struct test2::A (base)
// CHECK-NEXT:          0 |     int i1
// CHECK-NEXT:          4 |     double d
// CHECK-NEXT:         12 |   int i2
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=4]

} // namespace test2

namespace test3 {
#pragma align(natural)
struct A {
  int i1;
  double d;
};
#pragma align(reset)

struct B {
  struct A a;
  int i2;
};

int c = sizeof(B);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::A
// CHECK-NEXT:          0 |   int i1
// CHECK-NEXT:          8 |   double d
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::B
// CHECK-NEXT:          0 |   struct test3::A a
// CHECK-NEXT:          0 |     int i1
// CHECK-NEXT:          8 |     double d
// CHECK-NEXT:         16 |   int i2
// CHECK-NEXT:            | [sizeof=24, dsize=24, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=24, nvalign=4, preferrednvalign=8]

} // namespace test3

namespace test4 {
struct A {
  int i1;
  double d;
};

#pragma align(natural)
struct B {
  int i2;
  struct A a;
};
#pragma align(reset)

int d = sizeof(B);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::A
// CHECK-NEXT:          0 |   int i1
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::B
// CHECK-NEXT:          0 |   int i2
// CHECK-NEXT:          4 |   struct test4::A a
// CHECK-NEXT:          4 |     int i1
// CHECK-NEXT:          8 |     double d
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=4]

} // namespace test4
