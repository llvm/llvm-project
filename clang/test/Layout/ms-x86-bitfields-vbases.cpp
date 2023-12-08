// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>&1 \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

struct B0 { int a; };
struct B1 { int a; };

struct A : virtual B0 { char a : 1; };

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct A
// CHECK-NEXT:    0 |   (A vbtable pointer)
// CHECK-NEXT:    4:0-0 |   char a
// CHECK-NEXT:    8 |   struct B0 (virtual base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct A
// CHECK-X64-NEXT:    0 |   (A vbtable pointer)
// CHECK-X64-NEXT:    8:0-0 |   char a
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct B : virtual B0 { short a : 1; };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct B
// CHECK-NEXT:    0 |   (B vbtable pointer)
// CHECK-NEXT:    4:0-0 |   short a
// CHECK-NEXT:    8 |   struct B0 (virtual base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct B
// CHECK-X64-NEXT:    0 |   (B vbtable pointer)
// CHECK-X64-NEXT:    8:0-0 |   short a
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct C : virtual B0 { char a : 1; char : 0; };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct C
// CHECK-NEXT:    0 |   (C vbtable pointer)
// CHECK-NEXT:    4:0-0 |   char a
// CHECK-NEXT:    5:- |   char
// CHECK-NEXT:    8 |   struct B0 (virtual base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct C
// CHECK-X64-NEXT:    0 |   (C vbtable pointer)
// CHECK-X64-NEXT:    8:0-0 |   char a
// CHECK-X64-NEXT:    9:- |   char
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct D : virtual B0 { char a : 1; char b; };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct D
// CHECK-NEXT:    0 |   (D vbtable pointer)
// CHECK-NEXT:    4:0-0 |   char a
// CHECK-NEXT:    5 |   char b
// CHECK-NEXT:    8 |   struct B0 (virtual base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct D
// CHECK-X64-NEXT:    0 |   (D vbtable pointer)
// CHECK-X64-NEXT:    8:0-0 |   char a
// CHECK-X64-NEXT:    9 |   char b
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct E : virtual B0, virtual B1 { long long : 1; };
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct E
// CHECK-NEXT:    0 |   (E vbtable pointer)
// CHECK-NEXT:    8:0-0 |   long long
// CHECK-NEXT:   16 |   struct B0 (virtual base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |   struct B1 (virtual base)
// CHECK-NEXT:   20 |     int a
// CHECK-NEXT:      | [sizeof=24, align=8
// CHECK-NEXT:      |  nvsize=16, nvalign=8]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct E
// CHECK-X64-NEXT:    0 |   (E vbtable pointer)
// CHECK-X64-NEXT:    8:0-0 |   long long
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   20 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   20 |     int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)];
