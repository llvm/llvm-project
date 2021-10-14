// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -Wpacked \
// RUN:     -fdump-record-layouts -fsyntax-only -verify -x c++ < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -Wpacked \
// RUN:     -fdump-record-layouts -fsyntax-only -verify -x c++ < %s | \
// RUN:   FileCheck %s

struct A {
  double d;
};

struct B {
  char x[8];
};

struct [[gnu::packed]] C : B, A { // expected-warning{{packed attribute is unnecessary for 'C'}}
  char x alignas(4)[8];
};

int b = sizeof(C);

// CHECK:               0 | struct C
// CHECK-NEXT:          0 |   struct B (base)
// CHECK-NEXT:          0 |     char [8] x
// CHECK-NEXT:          8 |   struct A (base)
// CHECK-NEXT:          8 |     double d
// CHECK-NEXT:         16 |   char [8] x
// CHECK-NEXT:            | [sizeof=24, dsize=24, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=24, nvalign=4, preferrednvalign=4]
