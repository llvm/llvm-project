// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fxl-pragma-pack -verify -fsyntax-only -x c++ %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fxl-pragma-pack -verify -fsyntax-only -x c++ %s | \
// RUN:   FileCheck %s

namespace test1 {
#pragma align(natural)
#pragma pack(4)
#pragma pack(2)
struct A {
  int i;
  double d;
};

int a = sizeof(A);
#pragma pack()
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 4}}
#pragma pack(pop)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
struct B {
  int i;
  double d;
};
#pragma align(reset)

int b = sizeof(B);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=2, preferredalign=2,
// CHECK-NEXT:            |  nvsize=12, nvalign=2, preferrednvalign=2]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test1::B
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          8 |   double d
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

} // namespace test1

namespace test2 {
#pragma align(natural)
#pragma pack(2)
struct A {
  int i;
  double d;
};

int a = sizeof(A);
#pragma align(reset)

struct B {
  int i;
  double d;
};

int b = sizeof(B);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=2, preferredalign=2,
// CHECK-NEXT:            |  nvsize=12, nvalign=2, preferrednvalign=2]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test2::B
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]

} // namespace test2

namespace test3 {
#pragma pack(2)
#pragma align(natural)
struct A {
  double d;
};
#pragma align(reset)
#pragma pack(pop)

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::A
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

} // namespace test3

namespace test4 {
#pragma pack(2)
#pragma align(natural)
#pragma pack(pop)

struct A {
  int i;
  double d;
} a;
#pragma align(reset)
#pragma pack(pop)

int i = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          8 |   double d
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

} // namespace test4

namespace test5 {
#pragma align(power)
#pragma align(natural)
#pragma pack(2)
#pragma align(reset)
struct A {
  int i;
  double d;
};
#pragma align(reset)

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]

} // namespace test5

namespace test6 {
#pragma align(natural)
#pragma pack(0)    // expected-error {{expected #pragma pack parameter to be '1', '2', '4', '8', or '16'}}
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}

struct A {
  int i;
  double d;
} a;
#pragma align(reset)

int i = sizeof(a);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test6::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          8 |   double d
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

} // namespace test6

namespace test7 {
#pragma align = natural // expected-warning {{missing '(' after '#pragma align' - ignoring}}
#pragma align(reset)    // expected-warning {{#pragma options align=reset failed: stack empty}}
} // namespace test7

namespace test8 {
#pragma align(packed)
#pragma pack(2)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 2}}
struct A {
  int i;
  double d;
};
#pragma align(reset)

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test8::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=2, preferredalign=2,
// CHECK-NEXT:            |  nvsize=12, nvalign=2, preferrednvalign=2]

} // namespace test8

namespace test9 {
#pragma pack(push, r1, 2) // expected-error {{specifying an identifier within `#pragma pack` is not supported on this target}}
struct A {
  int i;
  double d;
};
#pragma pack(pop) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test9::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]

} // namespace test9

namespace test10 {
#pragma pack(2)
#pragma align(reset)
struct A {
  int i;
  double d;
};

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test10::A
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]

} // namespace test10
