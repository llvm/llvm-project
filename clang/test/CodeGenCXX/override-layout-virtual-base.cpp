// RUN: %clang_cc1 -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-layout-virtual-base.layout %s | FileCheck --check-prefix=SIMPLE  %s
// RUN: %clang_cc1 -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts -foverride-record-layout=%S/Inputs/override-layout-virtual-base.layout %s | FileCheck %s

struct S1 {
  int a;
};

struct S2 : virtual S1 {
  virtual void foo() {}
};

// SIMPLE: Type: struct S3
// SIMPLE:   FieldOffsets: [64]
struct S3 : S2 {
  char b;
};

struct S4 {
};

struct S5 : S4 {
  virtual void foo() {}
};

// CHECK:      *** Dumping AST Record Layout
// CHECK:               0 | struct S2
// CHECK-NEXT:          0 |   (S2 vftable pointer)
// CHECK-NEXT:          8 |   (S2 vbtable pointer)
// CHECK-NEXT:          8 |   struct S1 (virtual base)
// CHECK-NEXT:          8 |     int a
// CHECK-NEXT:            | [sizeof=8, align=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=8]
// CHECK:      *** Dumping AST Record Layout
// CHECK:               0 | struct S3
// CHECK-NEXT:          0 |   struct S2 (primary base)
// CHECK-NEXT:          0 |     (S2 vftable pointer)
// CHECK-NEXT:          8 |     (S2 vbtable pointer)
// CHECK-NEXT:          8 |   char b
// CHECK-NEXT:         16 |   struct S1 (virtual base)
// CHECK-NEXT:         16 |     int a
// CHECK-NEXT:            | [sizeof=24, align=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=8]
// CHECK:      *** Dumping AST Record Layout
// CHECK:               0 | struct S5
// CHECK-NEXT:          0 |   (S5 vftable pointer)
// CHECK-NEXT:          0 |   struct S4 (base) (empty)
// CHECK-NEXT:            | [sizeof=8, align=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=8]

void use_structs() {
  S1 s1s[sizeof(S1)];
  S2 s2s[sizeof(S2)];
  S3 s3s[sizeof(S3)];
  S4 s4s[sizeof(S4)];
  S5 s5s[sizeof(S5)];
}
