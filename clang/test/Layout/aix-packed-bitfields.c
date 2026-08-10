// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c %s | \
// RUN:   FileCheck --check-prefixes=CHECK,32BIT %s

// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | \
// RUN:   FileCheck --check-prefixes=CHECK,32BIT %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c %s | \
// RUN:   FileCheck --check-prefixes=CHECK,64BIT %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack -x c++ %s | \
// RUN:   FileCheck --check-prefixes=CHECK,64BIT %s

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
//
struct __attribute__((packed)) PackedAttr {
  char f1;
  int : 0;
  short : 3;
  char f4 : 2;
};

int e = sizeof(struct PackedAttr);
// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct PackedAttr
// CHECK-NEXT:      0 |   char f1
// CHECK-NEXT:    4:- |   int
// CHECK-NEXT:  4:0-2 |   short
// CHECK-NEXT:  4:3-4 |   char f4
// CHECK-NEXT:          sizeof=5, {{(dsize=5, )?}}align=1, preferredalign=1

#pragma pack(2)
struct __attribute__((packed)) PackedAttrAndPragma {
  char f1;
  long long : 0;
};
#pragma pack(pop)

int f = sizeof(struct PackedAttrAndPragma);
// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:      0 | struct PackedAttrAndPragma
// CHECK-NEXT:      0 |   char f1
// 32BIT-NEXT:    4:- |   long long
// 32BIT-NEXT:          sizeof=4, {{(dsize=4, )?}}align=1, preferredalign=1
// 64BIT-NEXT:    8:- |   long long
// 64BIT-NEXT:          sizeof=8, {{(dsize=8, )?}}align=1, preferredalign=1
