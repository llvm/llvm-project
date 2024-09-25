// RUN: %clang_cc1 -fno-rtti -triple i686-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -fno-rtti -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64 --strict-whitespace

extern "C" int printf(const char *fmt, ...);

struct B0 { int a; B0() : a(0xf00000B0) { printf("B0 = %p\n", this); } virtual void f() { printf("B0"); } };
struct B1 { int a; B1() : a(0xf00000B1) { printf("B1 = %p\n", this); } virtual void g() { printf("B1"); } };
struct B2 { int a; B2() : a(0xf00000B2) { printf("B1 = %p\n", this); } };
struct B0X { int a; B0X() : a(0xf00000B0) {} };
struct B1X { int a; B1X() : a(0xf00000B1) {} virtual void f() { printf("B0"); } };
struct B2X : virtual B1X { int a; B2X() : a(0xf00000B2) {} };

struct A : virtual B0 {
};

// CHECK-LABEL:   0 | struct A
// CHECK-NEXT:    0 |   (A vbtable pointer)
// CHECK-NEXT:    4 |   struct B0 (virtual base)
// CHECK-NEXT:    4 |     (B0 vftable pointer)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct A
// CHECK-X64-NEXT:    0 |   (A vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct B0 (virtual base)
// CHECK-X64-NEXT:    8 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct B : virtual B0 {
	virtual void f() { printf("B"); }
};

// CHECK-LABEL:   0 | struct B
// CHECK-NEXT:    0 |   (B vbtable pointer)
// CHECK-NEXT:    4 |   struct B0 (virtual base)
// CHECK-NEXT:    4 |     (B0 vftable pointer)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct B
// CHECK-X64-NEXT:    0 |   (B vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct B0 (virtual base)
// CHECK-X64-NEXT:    8 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct C : virtual B0 {
	virtual void g() { printf("A"); }
};

// CHECK-LABEL:   0 | struct C
// CHECK-NEXT:    0 |   (C vftable pointer)
// CHECK-NEXT:    4 |   (C vbtable pointer)
// CHECK-NEXT:    8 |   struct B0 (virtual base)
// CHECK-NEXT:    8 |     (B0 vftable pointer)
// CHECK-NEXT:   12 |     int a
// CHECK-NEXT:      | [sizeof=16, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct C
// CHECK-X64-NEXT:    0 |   (C vftable pointer)
// CHECK-X64-NEXT:    8 |   (C vbtable pointer)
// CHECK-X64-NEXT:   16 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   16 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:      | [sizeof=32, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct D : virtual B2, virtual B0 {
	virtual void f() { printf("D"); }
	virtual void g() { printf("D"); }
};

// CHECK-LABEL:   0 | struct D
// CHECK-NEXT:    0 |   (D vftable pointer)
// CHECK-NEXT:    4 |   (D vbtable pointer)
// CHECK-NEXT:    8 |   struct B2 (virtual base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:   12 |   struct B0 (virtual base)
// CHECK-NEXT:   12 |     (B0 vftable pointer)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct D
// CHECK-X64-NEXT:    0 |   (D vftable pointer)
// CHECK-X64-NEXT:    8 |   (D vbtable pointer)
// CHECK-X64-NEXT:   16 |   struct B2 (virtual base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   24 |   struct B0 (virtual base)
// CHECK-X64-NEXT:   24 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct E : B0, virtual B1 {
	virtual void f() { printf("E"); }
	virtual void g() { printf("E"); }
};

// CHECK-LABEL:   0 | struct E
// CHECK-NEXT:    0 |   struct B0 (primary base)
// CHECK-NEXT:    0 |     (B0 vftable pointer)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   (E vbtable pointer)
// CHECK-NEXT:   12 |   struct B1 (virtual base)
// CHECK-NEXT:   12 |     (B1 vftable pointer)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64-LABEL:   0 | struct E
// CHECK-X64-NEXT:    0 |   struct B0 (primary base)
// CHECK-X64-NEXT:    0 |     (B0 vftable pointer)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   (E vbtable pointer)
// CHECK-X64-NEXT:   24 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   24 |     (B1 vftable pointer)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct F : virtual B0, virtual B1 {
};

// CHECK-LABEL:   0 | struct F
// CHECK-NEXT:    0 |   (F vbtable pointer)
// CHECK-NEXT:    4 |   struct B0 (virtual base)
// CHECK-NEXT:    4 |     (B0 vftable pointer)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:   12 |   struct B1 (virtual base)
// CHECK-NEXT:   12 |     (B1 vftable pointer)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct F
// CHECK-X64-NEXT:    0 |   (F vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct B0 (virtual base)
// CHECK-X64-NEXT:    8 |     (B0 vftable pointer)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   24 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   24 |     (B1 vftable pointer)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct AX : B0X, B1X { int a; AX() : a(0xf000000A) {} virtual void f() { printf("A"); } };

// CHECK-LABEL:   0 | struct AX
// CHECK-NEXT:    0 |   struct B1X (primary base)
// CHECK-NEXT:    0 |     (B1X vftable pointer)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   struct B0X (base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:   12 |   int a
// CHECK-NEXT:      | [sizeof=16, align=4
// CHECK-NEXT:      |  nvsize=16, nvalign=4]
// CHECK-X64-LABEL:   0 | struct AX
// CHECK-X64-NEXT:    0 |   struct B1X (primary base)
// CHECK-X64-NEXT:    0 |     (B1X vftable pointer)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   struct B0X (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   20 |   int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct BX : B0X, B1X { int a; BX() : a(0xf000000B) {} virtual void g() { printf("B"); } };

// CHECK-LABEL:   0 | struct BX
// CHECK-NEXT:    0 |   struct B1X (primary base)
// CHECK-NEXT:    0 |     (B1X vftable pointer)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   struct B0X (base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:   12 |   int a
// CHECK-NEXT:      | [sizeof=16, align=4
// CHECK-NEXT:      |  nvsize=16, nvalign=4]
// CHECK-X64-LABEL:   0 | struct BX
// CHECK-X64-NEXT:    0 |   struct B1X (primary base)
// CHECK-X64-NEXT:    0 |     (B1X vftable pointer)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   struct B0X (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   20 |   int a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct CX : B0X, B2X { int a; CX() : a(0xf000000C) {} virtual void g() { printf("C"); } };

// CHECK-LABEL:   0 | struct CX
// CHECK-NEXT:    0 |   (CX vftable pointer)
// CHECK-NEXT:    4 |   struct B0X (base)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   struct B2X (base)
// CHECK-NEXT:    8 |     (B2X vbtable pointer)
// CHECK-NEXT:   12 |     int a
// CHECK-NEXT:   16 |   int a
// CHECK-NEXT:   20 |   struct B1X (virtual base)
// CHECK-NEXT:   20 |     (B1X vftable pointer)
// CHECK-NEXT:   24 |     int a
// CHECK-NEXT:      | [sizeof=28, align=4
// CHECK-NEXT:      |  nvsize=20, nvalign=4]
// CHECK-X64-LABEL:   0 | struct CX
// CHECK-X64-NEXT:    0 |   (CX vftable pointer)
// CHECK-X64-NEXT:    8 |   struct B0X (base)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   struct B2X (base)
// CHECK-X64-NEXT:   16 |     (B2X vbtable pointer)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   40 |   struct B1X (virtual base)
// CHECK-X64-NEXT:   40 |     (B1X vftable pointer)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:      | [sizeof=56, align=8
// CHECK-X64-NEXT:      |  nvsize=40, nvalign=8]

struct DX : virtual B1X { int a; DX() : a(0xf000000D) {} virtual void f() { printf("D"); } };

// CHECK-LABEL:   0 | struct DX
// CHECK-NEXT:    0 |   (DX vbtable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:    8 |   (vtordisp for vbase B1X)
// CHECK-NEXT:   12 |   struct B1X (virtual base)
// CHECK-NEXT:   12 |     (B1X vftable pointer)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct DX
// CHECK-X64-NEXT:    0 |   (DX vbtable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:   20 |   (vtordisp for vbase B1X)
// CHECK-X64-NEXT:   24 |   struct B1X (virtual base)
// CHECK-X64-NEXT:   24 |     (B1X vftable pointer)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct EX : virtual B1X { int a; EX() : a(0xf000000E) {} virtual void g() { printf("E"); } };

// CHECK-LABEL:   0 | struct EX
// CHECK-NEXT:    0 |   (EX vftable pointer)
// CHECK-NEXT:    4 |   (EX vbtable pointer)
// CHECK-NEXT:    8 |   int a
// CHECK-NEXT:   12 |   struct B1X (virtual base)
// CHECK-NEXT:   12 |     (B1X vftable pointer)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64-LABEL:   0 | struct EX
// CHECK-X64-NEXT:    0 |   (EX vftable pointer)
// CHECK-X64-NEXT:    8 |   (EX vbtable pointer)
// CHECK-X64-NEXT:   16 |   int a
// CHECK-X64-NEXT:   24 |   struct B1X (virtual base)
// CHECK-X64-NEXT:   24 |     (B1X vftable pointer)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct FX : virtual B1X { int a; FX() : a(0xf000000F) {} };

// CHECK-LABEL:   0 | struct FX
// CHECK-NEXT:    0 |   (FX vbtable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:    8 |   struct B1X (virtual base)
// CHECK-NEXT:    8 |     (B1X vftable pointer)
// CHECK-NEXT:   12 |     int a
// CHECK-NEXT:      | [sizeof=16, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct FX
// CHECK-X64-NEXT:    0 |   (FX vbtable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:   16 |   struct B1X (virtual base)
// CHECK-X64-NEXT:   16 |     (B1X vftable pointer)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:      | [sizeof=32, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)+
sizeof(E)+
sizeof(F)+
sizeof(AX)+
sizeof(BX)+
sizeof(CX)+
sizeof(DX)+
sizeof(EX)+
sizeof(FX)];
