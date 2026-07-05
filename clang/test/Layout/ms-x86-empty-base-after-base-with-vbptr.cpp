// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64


struct U { char a; };
struct V { };
struct W { };
struct X : virtual V { char a; };
struct Y : virtual V { char a; };
struct Z : Y { };

struct A : X, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct A
// CHECK-NEXT:    0 |   struct X (base)
// CHECK-NEXT:    0 |     (X vbtable pointer)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    9 |   struct W (base) (empty)
// CHECK-NEXT:    9 |   char a
// CHECK-NEXT:   12 |   struct V (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct A
// CHECK-X64-NEXT:    0 |   struct X (base)
// CHECK-X64-NEXT:    0 |     (X vbtable pointer)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   17 |   struct W (base) (empty)
// CHECK-X64-NEXT:   17 |   char a
// CHECK-X64-NEXT:   24 |   struct V (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct B : X, U, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct B
// CHECK-NEXT:    0 |   struct X (base)
// CHECK-NEXT:    0 |     (X vbtable pointer)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    8 |   struct U (base)
// CHECK-NEXT:    8 |     char a
// CHECK-NEXT:    9 |   struct W (base) (empty)
// CHECK-NEXT:    9 |   char a
// CHECK-NEXT:   12 |   struct V (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct B
// CHECK-X64-NEXT:    0 |   struct X (base)
// CHECK-X64-NEXT:    0 |     (X vbtable pointer)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   16 |   struct U (base)
// CHECK-X64-NEXT:   16 |     char a
// CHECK-X64-NEXT:   17 |   struct W (base) (empty)
// CHECK-X64-NEXT:   17 |   char a
// CHECK-X64-NEXT:   24 |   struct V (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct C : X, V, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct C
// CHECK-NEXT:    0 |   struct X (base)
// CHECK-NEXT:    0 |     (X vbtable pointer)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    9 |   struct V (base) (empty)
// CHECK-NEXT:   10 |   struct W (base) (empty)
// CHECK-NEXT:   10 |   char a
// CHECK-NEXT:   12 |   struct V (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct C
// CHECK-X64-NEXT:    0 |   struct X (base)
// CHECK-X64-NEXT:    0 |     (X vbtable pointer)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   17 |   struct V (base) (empty)
// CHECK-X64-NEXT:   18 |   struct W (base) (empty)
// CHECK-X64-NEXT:   18 |   char a
// CHECK-X64-NEXT:   24 |   struct V (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct D : X, U, V, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct D
// CHECK-NEXT:    0 |   struct X (base)
// CHECK-NEXT:    0 |     (X vbtable pointer)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    8 |   struct U (base)
// CHECK-NEXT:    8 |     char a
// CHECK-NEXT:    9 |   struct V (base) (empty)
// CHECK-NEXT:   10 |   struct W (base) (empty)
// CHECK-NEXT:   10 |   char a
// CHECK-NEXT:   12 |   struct V (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct D
// CHECK-X64-NEXT:    0 |   struct X (base)
// CHECK-X64-NEXT:    0 |     (X vbtable pointer)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   16 |   struct U (base)
// CHECK-X64-NEXT:   16 |     char a
// CHECK-X64-NEXT:   17 |   struct V (base) (empty)
// CHECK-X64-NEXT:   18 |   struct W (base) (empty)
// CHECK-X64-NEXT:   18 |   char a
// CHECK-X64-NEXT:   24 |   struct V (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct E : X, U, Y, V, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct E
// CHECK-NEXT:    0 |   struct X (base)
// CHECK-NEXT:    0 |     (X vbtable pointer)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    8 |   struct U (base)
// CHECK-NEXT:    8 |     char a
// CHECK-NEXT:   12 |   struct Y (base)
// CHECK-NEXT:   12 |     (Y vbtable pointer)
// CHECK-NEXT:   16 |     char a
// CHECK-NEXT:   21 |   struct V (base) (empty)
// CHECK-NEXT:   22 |   struct W (base) (empty)
// CHECK-NEXT:   22 |   char a
// CHECK-NEXT:   24 |   struct V (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=24, align=4
// CHECK-NEXT:      |  nvsize=24, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct E
// CHECK-X64-NEXT:    0 |   struct X (base)
// CHECK-X64-NEXT:    0 |     (X vbtable pointer)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   16 |   struct U (base)
// CHECK-X64-NEXT:   16 |     char a
// CHECK-X64-NEXT:   24 |   struct Y (base)
// CHECK-X64-NEXT:   24 |     (Y vbtable pointer)
// CHECK-X64-NEXT:   32 |     char a
// CHECK-X64-NEXT:   41 |   struct V (base) (empty)
// CHECK-X64-NEXT:   42 |   struct W (base) (empty)
// CHECK-X64-NEXT:   42 |   char a
// CHECK-X64-NEXT:   48 |   struct V (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=48, align=8
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=8]

struct F : Z, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct F
// CHECK-NEXT:    0 |   struct Z (base)
// CHECK-NEXT:    0 |     struct Y (base)
// CHECK-NEXT:    0 |       (Y vbtable pointer)
// CHECK-NEXT:    4 |       char a
// CHECK-NEXT:    9 |   struct W (base) (empty)
// CHECK-NEXT:    9 |   char a
// CHECK-NEXT:   12 |   struct V (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct F
// CHECK-X64-NEXT:    0 |   struct Z (base)
// CHECK-X64-NEXT:    0 |     struct Y (base)
// CHECK-X64-NEXT:    0 |       (Y vbtable pointer)
// CHECK-X64-NEXT:    8 |       char a
// CHECK-X64-NEXT:   17 |   struct W (base) (empty)
// CHECK-X64-NEXT:   17 |   char a
// CHECK-X64-NEXT:   24 |   struct V (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct G : X, W, Y, V  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct G
// CHECK-NEXT:    0 |   struct X (base)
// CHECK-NEXT:    0 |     (X vbtable pointer)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    9 |   struct W (base) (empty)
// CHECK-NEXT:   12 |   struct Y (base)
// CHECK-NEXT:   12 |     (Y vbtable pointer)
// CHECK-NEXT:   16 |     char a
// CHECK-NEXT:   21 |   struct V (base) (empty)
// CHECK-NEXT:   21 |   char a
// CHECK-NEXT:   24 |   struct V (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=24, align=4
// CHECK-NEXT:      |  nvsize=24, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct G
// CHECK-X64-NEXT:    0 |   struct X (base)
// CHECK-X64-NEXT:    0 |     (X vbtable pointer)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   17 |   struct W (base) (empty)
// CHECK-X64-NEXT:   24 |   struct Y (base)
// CHECK-X64-NEXT:   24 |     (Y vbtable pointer)
// CHECK-X64-NEXT:   32 |     char a
// CHECK-X64-NEXT:   41 |   struct V (base) (empty)
// CHECK-X64-NEXT:   41 |   char a
// CHECK-X64-NEXT:   48 |   struct V (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=48, align=8
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)+
sizeof(G)];
