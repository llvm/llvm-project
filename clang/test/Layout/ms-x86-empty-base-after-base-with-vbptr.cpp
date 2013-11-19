// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fdump-record-layouts -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -cxx-abi microsoft %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64


struct U { char a; };
struct V { };
struct W { };
struct X : virtual V { char a; };
struct Y : virtual V { char a; };
struct Z : Y { };

struct A : X, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct A
// CHECK:    0 |   struct X (base)
// CHECK:    0 |     (X vbtable pointer)
// CHECK:    4 |     char a
// CHECK:    9 |   struct W (base) (empty)
// CHECK:    9 |   char a
// CHECK:   12 |   struct V (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct A
// CHECK-X64:    0 |   struct X (base)
// CHECK-X64:    0 |     (X vbtable pointer)
// CHECK-X64:    8 |     char a
// CHECK-X64:   17 |   struct W (base) (empty)
// CHECK-X64:   17 |   char a
// CHECK-X64:   24 |   struct V (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct B : X, U, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct B
// CHECK:    0 |   struct X (base)
// CHECK:    0 |     (X vbtable pointer)
// CHECK:    4 |     char a
// CHECK:    8 |   struct U (base)
// CHECK:    8 |     char a
// CHECK:    9 |   struct W (base) (empty)
// CHECK:    9 |   char a
// CHECK:   12 |   struct V (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct B
// CHECK-X64:    0 |   struct X (base)
// CHECK-X64:    0 |     (X vbtable pointer)
// CHECK-X64:    8 |     char a
// CHECK-X64:   16 |   struct U (base)
// CHECK-X64:   16 |     char a
// CHECK-X64:   17 |   struct W (base) (empty)
// CHECK-X64:   17 |   char a
// CHECK-X64:   24 |   struct V (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct C : X, V, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct C
// CHECK:    0 |   struct X (base)
// CHECK:    0 |     (X vbtable pointer)
// CHECK:    4 |     char a
// CHECK:    9 |   struct V (base) (empty)
// CHECK:   10 |   struct W (base) (empty)
// CHECK:   10 |   char a
// CHECK:   12 |   struct V (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct C
// CHECK-X64:    0 |   struct X (base)
// CHECK-X64:    0 |     (X vbtable pointer)
// CHECK-X64:    8 |     char a
// CHECK-X64:   17 |   struct V (base) (empty)
// CHECK-X64:   18 |   struct W (base) (empty)
// CHECK-X64:   18 |   char a
// CHECK-X64:   24 |   struct V (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct D : X, U, V, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct D
// CHECK:    0 |   struct X (base)
// CHECK:    0 |     (X vbtable pointer)
// CHECK:    4 |     char a
// CHECK:    8 |   struct U (base)
// CHECK:    8 |     char a
// CHECK:    9 |   struct V (base) (empty)
// CHECK:   10 |   struct W (base) (empty)
// CHECK:   10 |   char a
// CHECK:   12 |   struct V (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct D
// CHECK-X64:    0 |   struct X (base)
// CHECK-X64:    0 |     (X vbtable pointer)
// CHECK-X64:    8 |     char a
// CHECK-X64:   16 |   struct U (base)
// CHECK-X64:   16 |     char a
// CHECK-X64:   17 |   struct V (base) (empty)
// CHECK-X64:   18 |   struct W (base) (empty)
// CHECK-X64:   18 |   char a
// CHECK-X64:   24 |   struct V (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct E : X, U, Y, V, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct E
// CHECK:    0 |   struct X (base)
// CHECK:    0 |     (X vbtable pointer)
// CHECK:    4 |     char a
// CHECK:    8 |   struct U (base)
// CHECK:    8 |     char a
// CHECK:   12 |   struct Y (base)
// CHECK:   12 |     (Y vbtable pointer)
// CHECK:   16 |     char a
// CHECK:   21 |   struct V (base) (empty)
// CHECK:   22 |   struct W (base) (empty)
// CHECK:   22 |   char a
// CHECK:   24 |   struct V (virtual base) (empty)
// CHECK:      | [sizeof=24, align=4
// CHECK:      |  nvsize=24, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct E
// CHECK-X64:    0 |   struct X (base)
// CHECK-X64:    0 |     (X vbtable pointer)
// CHECK-X64:    8 |     char a
// CHECK-X64:   16 |   struct U (base)
// CHECK-X64:   16 |     char a
// CHECK-X64:   24 |   struct Y (base)
// CHECK-X64:   24 |     (Y vbtable pointer)
// CHECK-X64:   32 |     char a
// CHECK-X64:   41 |   struct V (base) (empty)
// CHECK-X64:   42 |   struct W (base) (empty)
// CHECK-X64:   42 |   char a
// CHECK-X64:   48 |   struct V (virtual base) (empty)
// CHECK-X64:      | [sizeof=48, align=8
// CHECK-X64:      |  nvsize=48, nvalign=8]

struct F : Z, W  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct F
// CHECK:    0 |   struct Z (base)
// CHECK:    0 |     struct Y (base)
// CHECK:    0 |       (Y vbtable pointer)
// CHECK:    4 |       char a
// CHECK:    9 |   struct W (base) (empty)
// CHECK:    9 |   char a
// CHECK:   12 |   struct V (virtual base) (empty)
// CHECK:      | [sizeof=12, align=4
// CHECK:      |  nvsize=12, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct F
// CHECK-X64:    0 |   struct Z (base)
// CHECK-X64:    0 |     struct Y (base)
// CHECK-X64:    0 |       (Y vbtable pointer)
// CHECK-X64:    8 |       char a
// CHECK-X64:   17 |   struct W (base) (empty)
// CHECK-X64:   17 |   char a
// CHECK-X64:   24 |   struct V (virtual base) (empty)
// CHECK-X64:      | [sizeof=24, align=8
// CHECK-X64:      |  nvsize=24, nvalign=8]

struct G : X, W, Y, V  { char a; };

// CHECK: *** Dumping AST Record Layout
// CHECK:    0 | struct G
// CHECK:    0 |   struct X (base)
// CHECK:    0 |     (X vbtable pointer)
// CHECK:    4 |     char a
// CHECK:    9 |   struct W (base) (empty)
// CHECK:   12 |   struct Y (base)
// CHECK:   12 |     (Y vbtable pointer)
// CHECK:   16 |     char a
// CHECK:   21 |   struct V (base) (empty)
// CHECK:   21 |   char a
// CHECK:   24 |   struct V (virtual base) (empty)
// CHECK:      | [sizeof=24, align=4
// CHECK:      |  nvsize=24, nvalign=4]
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64:    0 | struct G
// CHECK-X64:    0 |   struct X (base)
// CHECK-X64:    0 |     (X vbtable pointer)
// CHECK-X64:    8 |     char a
// CHECK-X64:   17 |   struct W (base) (empty)
// CHECK-X64:   24 |   struct Y (base)
// CHECK-X64:   24 |     (Y vbtable pointer)
// CHECK-X64:   32 |     char a
// CHECK-X64:   41 |   struct V (base) (empty)
// CHECK-X64:   41 |   char a
// CHECK-X64:   48 |   struct V (virtual base) (empty)
// CHECK-X64:      | [sizeof=48, align=8
// CHECK-X64:      |  nvsize=48, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)+
sizeof(G)];
