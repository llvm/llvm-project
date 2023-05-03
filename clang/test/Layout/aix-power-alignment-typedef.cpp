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
// CHECK-NEXT:     0 |   Dbl x
// CHECK-NEXT:       | [sizeof=8, dsize=8, align=2, preferredalign=2,
// CHECK-NEXT:       |  nvsize=8, nvalign=2, preferrednvalign=2]

} // namespace test1

