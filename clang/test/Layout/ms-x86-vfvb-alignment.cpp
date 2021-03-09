// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>&1 \
// RUN:            | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64 --strict-whitespace

extern "C" int printf(const char *fmt, ...);

struct B0 { int a; B0() : a(0xf00000B0) {} };
struct B1 { char a; B1() : a(0xB1) {} };
struct B2 : virtual B1 { int a; B2() : a(0xf00000B2) {} };
struct B3 { __declspec(align(16)) int a; B3() : a(0xf00000B3) {} };
struct B4 : virtual B3 { int a; B4() : a(0xf00000B4) {} };
struct B5 { __declspec(align(32)) int a; B5() : a(0xf00000B5) {} };
struct B6 { int a; B6() : a(0xf00000B6) {} virtual void f() { printf("B6"); } };

struct A : B0, virtual B1 { __declspec(align(16)) int a; A() : a(0xf000000A) {} virtual void f() { printf("A"); } };

// CHECK-LABEL:   0 | struct A
// CHECK-NEXT:    0 |   (A vftable pointer)
// CHECK-NEXT:   16 |   struct B0 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |   (A vbtable pointer)
// CHECK-NEXT:   48 |   int a
// CHECK-NEXT:   64 |   struct B1 (virtual base)
// CHECK-NEXT:   64 |     char a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=64, nvalign=16]
// CHECK-X64-LABEL:   0 | struct A
// CHECK-X64-NEXT:    0 |   (A vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B0 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   24 |   (A vbtable pointer)
// CHECK-X64-NEXT:   48 |   int a
// CHECK-X64-NEXT:   64 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   64 |     char a
// CHECK-X64-NEXT:      | [sizeof=80, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct C : B4 {
  int a;
  C() : a(0xf000000C) {}
  virtual void f() { printf("C"); }
};

// CHECK-LABEL:   0 | struct C
// CHECK-NEXT:    0 |   (C vftable pointer)
// CHECK-NEXT:   16 |   struct B4 (base)
// CHECK-NEXT:   16 |     (B4 vbtable pointer)
// CHECK-NEXT:   20 |     int a
// CHECK-NEXT:   24 |   int a
// CHECK-NEXT:   32 |   struct B3 (virtual base)
// CHECK-NEXT:   32 |     int a
// CHECK-NEXT:      | [sizeof=48, align=16
// CHECK-NEXT:      |  nvsize=32, nvalign=16]
// CHECK-X64-LABEL:   0 | struct C
// CHECK-X64-NEXT:    0 |   (C vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B4 (base)
// CHECK-X64-NEXT:   16 |     (B4 vbtable pointer)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   48 |   struct B3 (virtual base)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct K {
  int a;
  K() : a(0xf0000013) {}
  virtual void f() { printf("K"); }
};

// CHECK-LABEL:   0 | struct K
// CHECK-NEXT:    0 |   (K vftable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct K
// CHECK-X64-NEXT:    0 |   (K vftable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct B : A, B2 { int a; B() : a(0xf000000B) {} virtual void f() { printf("B"); } };

// CHECK-LABEL:   0 | struct B{{$}}
// CHECK-NEXT:    0 |   struct A (primary base)
// CHECK-NEXT:    0 |     (A vftable pointer)
// CHECK-NEXT:   16 |     struct B0 (base)
// CHECK-NEXT:   16 |       int a
// CHECK-NEXT:   20 |     (A vbtable pointer)
// CHECK-NEXT:   48 |     int a
// CHECK-NEXT:   64 |   struct B2 (base)
// CHECK-NEXT:   64 |     (B2 vbtable pointer)
// CHECK-NEXT:   68 |     int a
// CHECK-NEXT:   72 |   int a
// CHECK-NEXT:   80 |   struct B1 (virtual base)
// CHECK-NEXT:   80 |     char a
// CHECK-NEXT:      | [sizeof=96, align=16
// CHECK-NEXT:      |  nvsize=80, nvalign=16]
// CHECK-X64-LABEL:   0 | struct B{{$}}
// CHECK-X64-NEXT:    0 |   struct A (primary base)
// CHECK-X64-NEXT:    0 |     (A vftable pointer)
// CHECK-X64-NEXT:   16 |     struct B0 (base)
// CHECK-X64-NEXT:   16 |       int a
// CHECK-X64-NEXT:   24 |     (A vbtable pointer)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:   64 |   struct B2 (base)
// CHECK-X64-NEXT:   64 |     (B2 vbtable pointer)
// CHECK-X64-NEXT:   72 |     int a
// CHECK-X64-NEXT:   80 |   int a
// CHECK-X64-NEXT:   96 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   96 |     char a
// CHECK-X64-NEXT:      | [sizeof=112, align=16
// CHECK-X64-NEXT:      |  nvsize=96, nvalign=16]

struct D : C { int a; D() : a(0xf000000D) {} virtual void f() { printf("D"); } };

// CHECK-LABEL:   0 | struct D
// CHECK-NEXT:    0 |   struct C (primary base)
// CHECK-NEXT:    0 |     (C vftable pointer)
// CHECK-NEXT:   16 |     struct B4 (base)
// CHECK-NEXT:   16 |       (B4 vbtable pointer)
// CHECK-NEXT:   20 |       int a
// CHECK-NEXT:   24 |     int a
// CHECK-NEXT:   32 |   int a
// CHECK-NEXT:   48 |   struct B3 (virtual base)
// CHECK-NEXT:   48 |     int a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct D
// CHECK-X64-NEXT:    0 |   struct C (primary base)
// CHECK-X64-NEXT:    0 |     (C vftable pointer)
// CHECK-X64-NEXT:   16 |     struct B4 (base)
// CHECK-X64-NEXT:   16 |       (B4 vbtable pointer)
// CHECK-X64-NEXT:   24 |       int a
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:   48 |   int a
// CHECK-X64-NEXT:   64 |   struct B3 (virtual base)
// CHECK-X64-NEXT:   64 |     int a
// CHECK-X64-NEXT:      | [sizeof=80, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct E : virtual C { int a; E() : a(0xf000000E) {} virtual void f() { printf("E"); } };

// CHECK-LABEL:   0 | struct E
// CHECK-NEXT:    0 |   (E vbtable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:   16 |   struct B3 (virtual base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   44 |   (vtordisp for vbase C)
// CHECK-NEXT:   48 |   struct C (virtual base)
// CHECK-NEXT:   48 |     (C vftable pointer)
// CHECK-NEXT:   64 |     struct B4 (base)
// CHECK-NEXT:   64 |       (B4 vbtable pointer)
// CHECK-NEXT:   68 |       int a
// CHECK-NEXT:   72 |     int a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=8, nvalign=16]
// CHECK-X64-LABEL:   0 | struct E
// CHECK-X64-NEXT:    0 |   (E vbtable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:   16 |   struct B3 (virtual base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   44 |   (vtordisp for vbase C)
// CHECK-X64-NEXT:   48 |   struct C (virtual base)
// CHECK-X64-NEXT:   48 |     (C vftable pointer)
// CHECK-X64-NEXT:   64 |     struct B4 (base)
// CHECK-X64-NEXT:   64 |       (B4 vbtable pointer)
// CHECK-X64-NEXT:   72 |       int a
// CHECK-X64-NEXT:   80 |     int a
// CHECK-X64-NEXT:      | [sizeof=96, align=16
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=16]

struct F : B3, virtual B0 { int a; F() : a(0xf000000F) {} virtual void f() { printf("F"); } };

// CHECK-LABEL:   0 | struct F
// CHECK-NEXT:    0 |   (F vftable pointer)
// CHECK-NEXT:   16 |   struct B3 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   32 |   (F vbtable pointer)
// CHECK-NEXT:   48 |   int a
// CHECK-NEXT:   64 |   struct B0 (virtual base)
// CHECK-NEXT:   64 |     int a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=64, nvalign=16]
// CHECK-X64-LABEL:   0 | struct F
// CHECK-X64-NEXT:    0 |   (F vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B3 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   32 |   (F vbtable pointer)
// CHECK-X64-NEXT:   48 |   int a
// CHECK-X64-NEXT:   64 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   64 |     int a
// CHECK-X64-NEXT:      | [sizeof=80, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct G : B2, B6, virtual B1 { int a; G() : a(0xf0000010) {} };

// CHECK-LABEL:   0 | struct G
// CHECK-NEXT:    0 |   struct B6 (primary base)
// CHECK-NEXT:    0 |     (B6 vftable pointer)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   struct B2 (base)
// CHECK-NEXT:    8 |     (B2 vbtable pointer)
// CHECK-NEXT:   12 |     int a
// CHECK-NEXT:   16 |   int a
// CHECK-NEXT:   20 |   struct B1 (virtual base)
// CHECK-NEXT:   20 |     char a
// CHECK-NEXT:      | [sizeof=21, align=4
// CHECK-NEXT:      |  nvsize=20, nvalign=4]
// CHECK-X64-LABEL:   0 | struct G
// CHECK-X64-NEXT:    0 |   struct B6 (primary base)
// CHECK-X64-NEXT:    0 |     (B6 vftable pointer)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   struct B2 (base)
// CHECK-X64-NEXT:   16 |     (B2 vbtable pointer)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   40 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   40 |     char a
// CHECK-X64-NEXT:      | [sizeof=48, align=8
// CHECK-X64-NEXT:      |  nvsize=40, nvalign=8]

struct H : B6, B2, virtual B1 { int a; H() : a(0xf0000011) {} };

// CHECK-LABEL:   0 | struct H
// CHECK-NEXT:    0 |   struct B6 (primary base)
// CHECK-NEXT:    0 |     (B6 vftable pointer)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   struct B2 (base)
// CHECK-NEXT:    8 |     (B2 vbtable pointer)
// CHECK-NEXT:   12 |     int a
// CHECK-NEXT:   16 |   int a
// CHECK-NEXT:   20 |   struct B1 (virtual base)
// CHECK-NEXT:   20 |     char a
// CHECK-NEXT:      | [sizeof=21, align=4
// CHECK-NEXT:      |  nvsize=20, nvalign=4]
// CHECK-X64-LABEL:   0 | struct H
// CHECK-X64-NEXT:    0 |   struct B6 (primary base)
// CHECK-X64-NEXT:    0 |     (B6 vftable pointer)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   struct B2 (base)
// CHECK-X64-NEXT:   16 |     (B2 vbtable pointer)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   40 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   40 |     char a
// CHECK-X64-NEXT:      | [sizeof=48, align=8
// CHECK-X64-NEXT:      |  nvsize=40, nvalign=8]

struct I : B0, virtual B1 { int a; int a1; __declspec(align(16)) int a2; I() : a(0xf0000011), a1(0xf0000011), a2(0xf0000011) {} };

// CHECK-LABEL:   0 | struct I
// CHECK-NEXT:    0 |   struct B0 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:    4 |   (I vbtable pointer)
// CHECK-NEXT:   20 |   int a
// CHECK-NEXT:   24 |   int a1
// CHECK-NEXT:   32 |   int a2
// CHECK-NEXT:   48 |   struct B1 (virtual base)
// CHECK-NEXT:   48 |     char a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct I
// CHECK-X64-NEXT:    0 |   struct B0 (base)
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64-NEXT:    8 |   (I vbtable pointer)
// CHECK-X64-NEXT:   20 |   int a
// CHECK-X64-NEXT:   24 |   int a1
// CHECK-X64-NEXT:   32 |   int a2
// CHECK-X64-NEXT:   48 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   48 |     char a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct J : B0, B3, virtual B1 { int a; int a1; J() : a(0xf0000012), a1(0xf0000012) {} };

// CHECK-LABEL:   0 | struct J
// CHECK-NEXT:    0 |   struct B0 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:   16 |   struct B3 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   32 |   (J vbtable pointer)
// CHECK-NEXT:   48 |   int a
// CHECK-NEXT:   52 |   int a1
// CHECK-NEXT:   64 |   struct B1 (virtual base)
// CHECK-NEXT:   64 |     char a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=64, nvalign=16]
// CHECK-X64-LABEL:   0 | struct J
// CHECK-X64-NEXT:    0 |   struct B0 (base)
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64-NEXT:   16 |   struct B3 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   32 |   (J vbtable pointer)
// CHECK-X64-NEXT:   48 |   int a
// CHECK-X64-NEXT:   52 |   int a1
// CHECK-X64-NEXT:   64 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   64 |     char a
// CHECK-X64-NEXT:      | [sizeof=80, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct L : virtual K { int a; L() : a(0xf0000014) {} virtual void g() { printf("L"); } };

// CHECK-LABEL:   0 | struct L
// CHECK-NEXT:    0 |   (L vftable pointer)
// CHECK-NEXT:    4 |   (L vbtable pointer)
// CHECK-NEXT:    8 |   int a
// CHECK-NEXT:   12 |   struct K (virtual base)
// CHECK-NEXT:   12 |     (K vftable pointer)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64-LABEL:   0 | struct L
// CHECK-X64-NEXT:    0 |   (L vftable pointer)
// CHECK-X64-NEXT:    8 |   (L vbtable pointer)
// CHECK-X64-NEXT:   16 |   int a
// CHECK-X64-NEXT:   24 |   struct K (virtual base)
// CHECK-X64-NEXT:   24 |     (K vftable pointer)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct M : virtual K { int a; M() : a(0xf0000015) {} virtual void f() { printf("M"); } };

// CHECK-LABEL:   0 | struct M
// CHECK-NEXT:    0 |   (M vbtable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:    8 |   (vtordisp for vbase K)
// CHECK-NEXT:   12 |   struct K (virtual base)
// CHECK-NEXT:   12 |     (K vftable pointer)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct M
// CHECK-X64-NEXT:    0 |   (M vbtable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:   20 |   (vtordisp for vbase K)
// CHECK-X64-NEXT:   24 |   struct K (virtual base)
// CHECK-X64-NEXT:   24 |     (K vftable pointer)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)+
sizeof(G)+
sizeof(H)+
sizeof(I)+
sizeof(J)+
sizeof(K)+
sizeof(L)+
sizeof(M)];
