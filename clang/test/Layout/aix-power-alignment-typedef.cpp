// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts %s | \
// RUN:   FileCheck %s

namespace test1 {
typedef double __attribute__((__aligned__(2))) Dbl;
struct A {
  Dbl x;
};

int b = sizeof(A);

// CHECK:          0 | struct test1::A
// CHECK-NEXT:     0 |   test1::Dbl x
// CHECK-NEXT:       | [sizeof=8, dsize=8, align=2, preferredalign=2,
// CHECK-NEXT:       |  nvsize=8, nvalign=2, preferrednvalign=2]

} // namespace test1

namespace test2 {
typedef double Dbl __attribute__((__aligned__(2)));
typedef Dbl DblArr[];

union U {
  DblArr da;
  char x;
};

int x = sizeof(U);

// CHECK:          0 | union test2::U
// CHECK-NEXT:     0 |   test2::DblArr da
// CHECK-NEXT:     0 |   char x
// CHECK-NEXT:       | [sizeof=2, dsize=2, align=2, preferredalign=2,
// CHECK-NEXT:       |  nvsize=2, nvalign=2, preferrednvalign=2]

} // namespace test2

namespace test3 {
typedef double DblArr[] __attribute__((__aligned__(2)));

union U {
  DblArr da;
  char x;
};

int x = sizeof(U);

// CHECK:          0 | union test3::U
// CHECK-NEXT:     0 |   test3::DblArr da
// CHECK-NEXT:     0 |   char x
// CHECK-NEXT:       | [sizeof=2, dsize=2, align=2, preferredalign=2,
// CHECK-NEXT:       |  nvsize=2, nvalign=2, preferrednvalign=2]

} // namespace test3

namespace test4 {
typedef double Dbl __attribute__((__aligned__(2)));

union U {
  Dbl DblArr[];
  char x;
};

int x = sizeof(U);

// CHECK:          0 | union test4::U
// CHECK-NEXT:     0 |   test4::Dbl [] DblArr
// CHECK-NEXT:     0 |   char x
// CHECK-NEXT:       | [sizeof=2, dsize=2, align=2, preferredalign=2,
// CHECK-NEXT:       |  nvsize=2, nvalign=2, preferrednvalign=2]

} // namespace test4
