// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

extern "C" int printf(const char *fmt, ...);

struct B0 { char a; B0() : a(0xB0) {} };
struct __declspec(align(1)) B1 {};

struct A : virtual B0 {};

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct A
// CHECK-NEXT:    0 |   (A vbtable pointer)
// CHECK-NEXT:    4 |   struct B0 (virtual base)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:      | [sizeof=5, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct A
// CHECK-X64-NEXT:    0 |   (A vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct B0 (virtual base)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct __declspec(align(1)) B : virtual B0 {};

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct B
// CHECK-NEXT:    0 |   (B vbtable pointer)
// CHECK-NEXT:    4 |   struct B0 (virtual base)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct B
// CHECK-X64-NEXT:    0 |   (B vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct B0 (virtual base)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct C : virtual B0 { int a; C() : a(0xC) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct C
// CHECK-NEXT:    0 |   (C vbtable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:    8 |   struct B0 (virtual base)
// CHECK-NEXT:    8 |     char a
// CHECK-NEXT:      | [sizeof=9, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct C
// CHECK-X64-NEXT:    0 |   (C vbtable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     char a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct D : virtual B0 { __declspec(align(1)) int a; D() : a(0xD) {} };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct D
// CHECK-NEXT:    0 |   (D vbtable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:    8 |   struct B0 (virtual base)
// CHECK-NEXT:    8 |     char a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct D
// CHECK-X64-NEXT:    0 |   (D vbtable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     char a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct E : virtual B0, virtual B1 {};

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct E
// CHECK-NEXT:    0 |   (E vbtable pointer)
// CHECK-NEXT:    4 |   struct B0 (virtual base)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    5 |   struct B1 (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct E
// CHECK-X64-NEXT:    0 |   (E vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct B0 (virtual base)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:    9 |   struct B1 (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct F { char a; virtual ~F(); };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct F
// CHECK-NEXT:    0 |   (F vftable pointer)
// CHECK-NEXT:    4 |   char a
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct F
// CHECK-X64-NEXT:    0 |   (F vftable pointer)
// CHECK-X64-NEXT:    8 |   char a
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)];
