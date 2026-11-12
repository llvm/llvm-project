// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK32 %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only %s | \
// RUN:   FileCheck --check-prefixes=CHECK,CHECK64 %s

namespace test1 {
struct A {
  double d1;
  virtual void boo() {}
};

struct B {
  double d2;
  A a;
};

struct C : public A {
  double d3;
};

int i = sizeof(B);
int j = sizeof(C);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:            0 | struct test1::A
// CHECK-NEXT:            0 |   (A vtable pointer)
// CHECK32-NEXT:          4 |   double d1
// CHECK32-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK32-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]
// CHECK64-NEXT:          8 |   double d1
// CHECK64-NEXT:            | [sizeof=16, dsize=16, align=8, preferredalign=8,
// CHECK64-NEXT:            |  nvsize=16, nvalign=8, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:            0 | struct test1::B
// CHECK-NEXT:            0 |   double d2
// CHECK-NEXT:            8 |   struct test1::A a
// CHECK-NEXT:            8 |     (A vtable pointer)
// CHECK32-NEXT:         12 |     double d1
// CHECK32-NEXT:            | [sizeof=24, dsize=20, align=4, preferredalign=8,
// CHECK32-NEXT:            |  nvsize=20, nvalign=4, preferrednvalign=8]
// CHECK64-NEXT:         16 |     double d1
// CHECK64-NEXT:            | [sizeof=24, dsize=24, align=8, preferredalign=8,
// CHECK64-NEXT:            |  nvsize=24, nvalign=8, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:            0 | struct test1::C
// CHECK-NEXT:            0 |   struct test1::A (primary base)
// CHECK-NEXT:            0 |     (A vtable pointer)
// CHECK32-NEXT:          4 |     double d1
// CHECK32-NEXT:         12 |   double d3
// CHECK32-NEXT:            | [sizeof=20, dsize=20, align=4, preferredalign=4,
// CHECK32-NEXT:            |  nvsize=20, nvalign=4, preferrednvalign=4]
// CHECK64-NEXT:          8 |     double d1
// CHECK64-NEXT:         16 |   double d3
// CHECK64-NEXT:            | [sizeof=24, dsize=24, align=8, preferredalign=8,
// CHECK64-NEXT:            |  nvsize=24, nvalign=8, preferrednvalign=8]

} // namespace test1

namespace test2 {
struct A {
  long long l1;
};

struct B : public virtual A {
  double d2;
};

#pragma pack(2)
struct C : public virtual A {
  double __attribute__((aligned(4))) d3;
};

int i = sizeof(B);
int j = sizeof(C);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:            0 | struct test2::A
// CHECK-NEXT:            0 |   long long l1
// CHECK-NEXT:              | [sizeof=8, dsize=8, align=8, preferredalign=8,
// CHECK-NEXT:              |  nvsize=8, nvalign=8, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:            0 | struct test2::B
// CHECK-NEXT:            0 |   (B vtable pointer)
// CHECK32-NEXT:          4 |   double d2
// CHECK64-NEXT:          8 |   double d2
// CHECK-NEXT:           16 |   struct test2::A (virtual base)
// CHECK-NEXT:           16 |     long long l1
// CHECK-NEXT:              | [sizeof=24, dsize=24, align=8, preferredalign=8,
// CHECK32-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]
// CHECK64-NEXT:            |  nvsize=16, nvalign=8, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:            0 | struct test2::C
// CHECK-NEXT:            0 |   (C vtable pointer)
// CHECK32-NEXT:          4 |   double d3
// CHECK32-NEXT:         12 |   struct test2::A (virtual base)
// CHECK32-NEXT:         12 |     long long l1
// CHECK32-NEXT:            | [sizeof=20, dsize=20, align=2, preferredalign=2,
// CHECK32-NEXT:            |  nvsize=12, nvalign=2, preferrednvalign=2]
// CHECK64-NEXT:          8 |   double d3
// CHECK64-NEXT:         16 |   struct test2::A (virtual base)
// CHECK64-NEXT:         16 |     long long l1
// CHECK64-NEXT:            | [sizeof=24, dsize=24, align=2, preferredalign=2,
// CHECK64-NEXT:            |  nvsize=16, nvalign=2, preferrednvalign=2]

} // namespace test2
