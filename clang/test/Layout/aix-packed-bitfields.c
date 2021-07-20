// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c %s | FileCheck  %s

// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | FileCheck %s
//
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c %s | FileCheck  %s
//
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | FileCheck %s

struct A {
  int a1 : 30;
  int a2 : 30;
  int a3 : 4;
};

int a = sizeof(struct A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct A
// CHECK-NEXT: 0:0-29 |   int a1
// CHECK-NEXT: 4:0-29 |   int a2
// CHECK-NEXT:  8:0-3 |   int a3
// CHECK-NEXT:          sizeof=12, {{(dsize=12, )?}}align=4, preferredalign=4

#pragma align(packed)
struct AlignPacked {
  int a1 : 30;
  int a2 : 30;
  int a3 : 4;
};
#pragma align(reset)

int b = sizeof(struct AlignPacked);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct AlignPacked
// CHECK-NEXT: 0:0-29 |   int a1
// CHECK-NEXT: 3:6-35 |   int a2
// CHECK-NEXT:  7:4-7 |   int a3
// CHECK-NEXT:          sizeof=8, {{(dsize=8, )?}}align=1, preferredalign=1

#pragma pack(1)
struct Pack1 {
  int a1 : 30;
  int a2 : 30;
  int a3 : 4;
};
#pragma pack(pop)

int c = sizeof(struct Pack1);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct Pack1
// CHECK-NEXT: 0:0-29 |   int a1
// CHECK-NEXT: 3:6-35 |   int a2
// CHECK-NEXT:  7:4-7 |   int a3
// CHECK-NEXT:          sizeof=8, {{(dsize=8, )?}}align=1, preferredalign=1

#pragma pack(2)
struct Pack2 {
  int a1 : 30;
  int a2 : 30;
  int a3 : 4;
};
#pragma pack(pop)

int d = sizeof(struct Pack2);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct Pack2
// CHECK-NEXT: 0:0-29 |   int a1
// CHECK-NEXT: 3:6-35 |   int a2
// CHECK-NEXT:  7:4-7 |   int a3
// CHECK-NEXT:          sizeof=8, {{(dsize=8, )?}}align=2, preferredalign=2
