// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -Wpacked \
// RUN:     -fdump-record-layouts -fsyntax-only -verify -x c++ < %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -Wpacked \
// RUN:     -fdump-record-layouts -fsyntax-only -verify -x c++ < %s | \
// RUN:   FileCheck %s

// expected-no-diagnostics

struct [[gnu::packed]] Q {
  double x [[gnu::aligned(4)]];
};

struct QQ : Q { char x; };

int a = sizeof(QQ);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct Q
// CHECK-NEXT:          0 |   double x
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=4]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct QQ
// CHECK-NEXT:          0 |   struct Q (base)
// CHECK-NEXT:          0 |     double x
// CHECK-NEXT:          8 |   char x
// CHECK-NEXT:            | [sizeof=12, dsize=9, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=9, nvalign=4, preferrednvalign=4]
