// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | \
// RUN:   FileCheck --check-prefixes=CHECK %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | \
// RUN:   FileCheck --check-prefixes=CHECK %s

struct A {
  bool b : 3;
};

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct A
// CHECK-NEXT:      0:0-2 |   _Bool b
// CHECK-NEXT:            | [sizeof=4, dsize=4, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4, preferrednvalign=4]

enum class Bool : bool { False = 0,
                         True = 1 };

struct B {
  Bool b : 1;
};

int b = sizeof(B);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct B
// CHECK-NEXT:      0:0-0 |   Bool b
// CHECK-NEXT:            | [sizeof=4, dsize=4, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4, preferrednvalign=4]

enum LL : unsigned long long { val = 1 };
