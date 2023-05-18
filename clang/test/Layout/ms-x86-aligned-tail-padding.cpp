// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64 --strict-whitespace

extern "C" int printf(const char *fmt, ...);

struct B0 {
	int a;
	B0() : a(0xf00000B0) {}
};
struct __declspec(align(16)) B1 {
	int a;
	B1() : a(0xf00000B1) {}
};
struct B2 {
	__declspec(align(16)) int a;
	B2() : a(0xf00000B2) {}
};
struct __declspec(align(16)) B3 {
	long long a1;
	int a;
	B3() : a(0xf00000B3), a1(0xf00000B3f00000B3ll) {}
};
struct V {
	char a;
	V() : a(0X11) {}
};
struct __declspec(align(32)) A16 {};
struct V1 : A16 { virtual void f() {} };
struct V2 {
	long long a;
	int a1;
	V2() : a(0xf0000011f0000011ll), a1(0xf0000011) {}
};
struct V3 {
	int a;
	V3() : a(0xf0000022) {}
};
struct __declspec(align(16)) A16X {
};
struct __declspec(align(16)) B0X {
	int a, a1;
	B0X() : a(0xf00000B0), a1(0xf00000B0) {}
};
struct B1X {
	int a;
	B1X() : a(0xf00000B1) {}
};
struct B2X {
	int a;
	B2X() : a(0xf00000B2) {}
};
struct __declspec(align(16)) B3X {
	int a;
	B3X() : a(0xf00000B3) {}
	virtual void g() {}
};
struct B4X : A16X {
	int a, a1;
	B4X() : a(0xf00000B4), a1(0xf00000B4) {}
};
struct B5X : virtual A16X {
	int a, a1;
	B5X() : a(0xf00000B5), a1(0xf00000B5) {}
};
struct B6X {
	int a;
	B6X() : a(0xf00000B6) {}
};

struct A : B1, B0, B2, virtual V {
	int a;
	A() : a(0xf000000A) {}
};

// CHECK-LABEL:   0 | struct A{{$}}
// CHECK-NEXT:    0 |   struct B1 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:    4 |   struct B0 (base)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:   16 |   struct B2 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   32 |   (A vbtable pointer)
// CHECK-NEXT:   48 |   int a
// CHECK-NEXT:   64 |   struct V (virtual base)
// CHECK-NEXT:   64 |     char a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=64, nvalign=16]
// CHECK-X64-LABEL:   0 | struct A{{$}}
// CHECK-X64-NEXT:    0 |   struct B1 (base)
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64-NEXT:    4 |   struct B0 (base)
// CHECK-X64-NEXT:    4 |     int a
// CHECK-X64-NEXT:   16 |   struct B2 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   32 |   (A vbtable pointer)
// CHECK-X64-NEXT:   48 |   int a
// CHECK-X64-NEXT:   64 |   struct V (virtual base)
// CHECK-X64-NEXT:   64 |     char a
// CHECK-X64-NEXT:      | [sizeof=80, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct B : B2, B0, B1, virtual V {
	int a;
	B() : a(0xf000000B) {}
};

// CHECK-LABEL:   0 | struct B{{$}}
// CHECK-NEXT:    0 |   struct B2 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:   16 |   struct B0 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   32 |   struct B1 (base)
// CHECK-NEXT:   32 |     int a
// CHECK-NEXT:   36 |   (B vbtable pointer)
// CHECK-NEXT:   52 |   int a
// CHECK-NEXT:   64 |   struct V (virtual base)
// CHECK-NEXT:   64 |     char a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=64, nvalign=16]
// CHECK-X64-LABEL:   0 | struct B{{$}}
// CHECK-X64-NEXT:    0 |   struct B2 (base)
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64-NEXT:   16 |   struct B0 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   32 |   struct B1 (base)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:   40 |   (B vbtable pointer)
// CHECK-X64-NEXT:   52 |   int a
// CHECK-X64-NEXT:   64 |   struct V (virtual base)
// CHECK-X64-NEXT:   64 |     char a
// CHECK-X64-NEXT:      | [sizeof=80, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct C : B1, B0, virtual V {
	int a;
	long long a1;
	C() : a(0xf000000C), a1(0xf000000Cf000000Cll) {}
};

// CHECK-LABEL:   0 | struct C{{$}}
// CHECK-NEXT:    0 |   struct B1 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:    4 |   struct B0 (base)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   (C vbtable pointer)
// CHECK-NEXT:   24 |   int a
// CHECK-NEXT:   32 |   long long a1
// CHECK-NEXT:   48 |   struct V (virtual base)
// CHECK-NEXT:   48 |     char a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct C{{$}}
// CHECK-X64-NEXT:    0 |   struct B1 (base)
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64-NEXT:    4 |   struct B0 (base)
// CHECK-X64-NEXT:    4 |     int a
// CHECK-X64-NEXT:    8 |   (C vbtable pointer)
// CHECK-X64-NEXT:   24 |   int a
// CHECK-X64-NEXT:   32 |   long long a1
// CHECK-X64-NEXT:   48 |   struct V (virtual base)
// CHECK-X64-NEXT:   48 |     char a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct D : B2, B0, virtual V {
	int a;
	D() : a(0xf000000D) {}
};

// CHECK-LABEL:   0 | struct D{{$}}
// CHECK-NEXT:    0 |   struct B2 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:   16 |   struct B0 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |   (D vbtable pointer)
// CHECK-NEXT:   36 |   int a
// CHECK-NEXT:   48 |   struct V (virtual base)
// CHECK-NEXT:   48 |     char a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct D{{$}}
// CHECK-X64-NEXT:    0 |   struct B2 (base)
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64-NEXT:   16 |   struct B0 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   24 |   (D vbtable pointer)
// CHECK-X64-NEXT:   36 |   int a
// CHECK-X64-NEXT:   48 |   struct V (virtual base)
// CHECK-X64-NEXT:   48 |     char a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct E : B3, B0, virtual V {
	int a;
	E() : a(0xf000000E) {}
};

// CHECK-LABEL:   0 | struct E{{$}}
// CHECK-NEXT:    0 |   struct B3 (base)
// CHECK-NEXT:    0 |     long long a1
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:   16 |   struct B0 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |   (E vbtable pointer)
// CHECK-NEXT:   36 |   int a
// CHECK-NEXT:   48 |   struct V (virtual base)
// CHECK-NEXT:   48 |     char a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct E{{$}}
// CHECK-X64-NEXT:    0 |   struct B3 (base)
// CHECK-X64-NEXT:    0 |     long long a1
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   struct B0 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   24 |   (E vbtable pointer)
// CHECK-X64-NEXT:   36 |   int a
// CHECK-X64-NEXT:   48 |   struct V (virtual base)
// CHECK-X64-NEXT:   48 |     char a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct F : B0, virtual V1 {
	__declspec(align(16)) int a;
	F() : a(0xf000000F) {}
	virtual void f() {}
};

// CHECK-LABEL:   0 | struct F{{$}}
// CHECK-NEXT:    0 |   struct B0 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:    4 |   (F vbtable pointer)
// CHECK-NEXT:   32 |   int a
// CHECK-NEXT:   92 |   (vtordisp for vbase V1)
// CHECK-NEXT:   96 |   struct V1 (virtual base)
// CHECK-NEXT:   96 |     (V1 vftable pointer)
// CHECK-NEXT:  128 |     struct A16 (base) (empty)
// CHECK-NEXT:      | [sizeof=128, align=32
// CHECK-NEXT:      |  nvsize=48, nvalign=32]
// CHECK-X64-LABEL:   0 | struct F{{$}}
// CHECK-X64-NEXT:    0 |   struct B0 (base)
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64-NEXT:    8 |   (F vbtable pointer)
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   92 |   (vtordisp for vbase V1)
// CHECK-X64-NEXT:   96 |   struct V1 (virtual base)
// CHECK-X64-NEXT:   96 |     (V1 vftable pointer)
// CHECK-X64-NEXT:  128 |     struct A16 (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=128, align=32
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=32]

struct G : virtual V2, virtual V3 {
	int a;
	G() : a(0xf0000001) {}
};

// CHECK-LABEL:   0 | struct G{{$}}
// CHECK-NEXT:    0 |   (G vbtable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:    8 |   struct V2 (virtual base)
// CHECK-NEXT:    8 |     long long a
// CHECK-NEXT:   16 |     int a1
// CHECK-NEXT:   24 |   struct V3 (virtual base)
// CHECK-NEXT:   24 |     int a
// CHECK-NEXT:      | [sizeof=28, align=8
// CHECK-NEXT:      |  nvsize=8, nvalign=8]
// CHECK-X64-LABEL:   0 | struct G{{$}}
// CHECK-X64-NEXT:    0 |   (G vbtable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:   16 |   struct V2 (virtual base)
// CHECK-X64-NEXT:   16 |     long long a
// CHECK-X64-NEXT:   24 |     int a1
// CHECK-X64-NEXT:   32 |   struct V3 (virtual base)
// CHECK-X64-NEXT:   32 |     int a
// CHECK-X64-NEXT:      | [sizeof=40, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct H {
	__declspec(align(16)) int a;
	int b;
	H() : a(0xf0000010), b(0xf0000010) {}
};

// CHECK-LABEL:   0 | struct H{{$}}
// CHECK-NEXT:    0 |   int a
// CHECK-NEXT:    4 |   int b
// CHECK-NEXT:      | [sizeof=16, align=16
// CHECK-NEXT:      |  nvsize=16, nvalign=16]
// CHECK-X64-LABEL:   0 | struct H{{$}}
// CHECK-X64-NEXT:    0 |   int a
// CHECK-X64-NEXT:    4 |   int b
// CHECK-X64-NEXT:      | [sizeof=16, align=16
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=16]

struct I {
	B2 a;
	int b;
	I() : b(0xf0000010) {}
};

// CHECK-LABEL:   0 | struct I{{$}}
// CHECK-NEXT:    0 |   struct B2 a
// CHECK-NEXT:    0 |     int a
// CHECK:        16 |   int b
// CHECK-NEXT:      | [sizeof=32, align=16
// CHECK-NEXT:      |  nvsize=32, nvalign=16]
// CHECK-X64-LABEL:   0 | struct I{{$}}
// CHECK-X64-NEXT:    0 |   struct B2 a
// CHECK-X64-NEXT:    0 |     int a
// CHECK-X64:        16 |   int b
// CHECK-X64-NEXT:      | [sizeof=32, align=16
// CHECK-X64-NEXT:      |  nvsize=32, nvalign=16]

struct AX : B0X, virtual B2X, virtual B6X, virtual B3X {
	int a;
	AX() : a(0xf000000A) {}
	virtual void f() {}
	virtual void g() {}
};

// CHECK-LABEL:   0 | struct AX{{$}}
// CHECK-NEXT:    0 |   (AX vftable pointer)
// CHECK-NEXT:   16 |   struct B0X (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |     int a1
// CHECK-NEXT:   24 |   (AX vbtable pointer)
// CHECK-NEXT:   40 |   int a
// CHECK-NEXT:   48 |   struct B2X (virtual base)
// CHECK-NEXT:   48 |     int a
// CHECK-NEXT:   52 |   struct B6X (virtual base)
// CHECK-NEXT:   52 |     int a
// CHECK-NEXT:   76 |   (vtordisp for vbase B3X)
// CHECK-NEXT:   80 |   struct B3X (virtual base)
// CHECK-NEXT:   80 |     (B3X vftable pointer)
// CHECK-NEXT:   84 |     int a
// CHECK-NEXT:      | [sizeof=96, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct AX{{$}}
// CHECK-X64-NEXT:    0 |   (AX vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B0X (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   20 |     int a1
// CHECK-X64-NEXT:   24 |   (AX vbtable pointer)
// CHECK-X64-NEXT:   40 |   int a
// CHECK-X64-NEXT:   48 |   struct B2X (virtual base)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:   52 |   struct B6X (virtual base)
// CHECK-X64-NEXT:   52 |     int a
// CHECK-X64-NEXT:   76 |   (vtordisp for vbase B3X)
// CHECK-X64-NEXT:   80 |   struct B3X (virtual base)
// CHECK-X64-NEXT:   80 |     (B3X vftable pointer)
// CHECK-X64-NEXT:   88 |     int a
// CHECK-X64-NEXT:      | [sizeof=96, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct BX : B4X, virtual B2X, virtual B6X, virtual B3X {
	int a;
	BX() : a(0xf000000B) {}
	virtual void f() {}
	virtual void g() {}
};

// CHECK-LABEL:   0 | struct BX{{$}}
// CHECK-NEXT:    0 |   (BX vftable pointer)
// CHECK-NEXT:   16 |   struct B4X (base)
// CHECK-NEXT:   16 |     struct A16X (base) (empty)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |     int a1
// CHECK-NEXT:   32 |   (BX vbtable pointer)
// CHECK-NEXT:   48 |   int a
// CHECK-NEXT:   64 |   struct B2X (virtual base)
// CHECK-NEXT:   64 |     int a
// CHECK-NEXT:   68 |   struct B6X (virtual base)
// CHECK-NEXT:   68 |     int a
// CHECK-NEXT:   92 |   (vtordisp for vbase B3X)
// CHECK-NEXT:   96 |   struct B3X (virtual base)
// CHECK-NEXT:   96 |     (B3X vftable pointer)
// CHECK-NEXT:  100 |     int a
// CHECK-NEXT:      | [sizeof=112, align=16
// CHECK-NEXT:      |  nvsize=64, nvalign=16]
// CHECK-X64-LABEL:   0 | struct BX{{$}}
// CHECK-X64-NEXT:    0 |   (BX vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B4X (base)
// CHECK-X64-NEXT:   16 |     struct A16X (base) (empty)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   20 |     int a1
// CHECK-X64-NEXT:   32 |   (BX vbtable pointer)
// CHECK-X64-NEXT:   48 |   int a
// CHECK-X64-NEXT:   64 |   struct B2X (virtual base)
// CHECK-X64-NEXT:   64 |     int a
// CHECK-X64-NEXT:   68 |   struct B6X (virtual base)
// CHECK-X64-NEXT:   68 |     int a
// CHECK-X64-NEXT:   92 |   (vtordisp for vbase B3X)
// CHECK-X64-NEXT:   96 |   struct B3X (virtual base)
// CHECK-X64-NEXT:   96 |     (B3X vftable pointer)
// CHECK-X64-NEXT:  104 |     int a
// CHECK-X64-NEXT:      | [sizeof=112, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct CX : B5X, virtual B2X, virtual B6X, virtual B3X {
	int a;
	CX() : a(0xf000000C) {}
	virtual void f() {}
	virtual void g() {}
};

// CHECK-LABEL:   0 | struct CX{{$}}
// CHECK-NEXT:    0 |   (CX vftable pointer)
// CHECK-NEXT:   16 |   struct B5X (base)
// CHECK-NEXT:   16 |     (B5X vbtable pointer)
// CHECK-NEXT:   20 |     int a
// CHECK-NEXT:   24 |     int a1
// CHECK-NEXT:   28 |   int a
// CHECK-NEXT:   32 |   struct A16X (virtual base) (empty)
// CHECK-NEXT:   32 |   struct B2X (virtual base)
// CHECK-NEXT:   32 |     int a
// CHECK-NEXT:   36 |   struct B6X (virtual base)
// CHECK-NEXT:   36 |     int a
// CHECK-NEXT:   60 |   (vtordisp for vbase B3X)
// CHECK-NEXT:   64 |   struct B3X (virtual base)
// CHECK-NEXT:   64 |     (B3X vftable pointer)
// CHECK-NEXT:   68 |     int a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=32, nvalign=16]
// CHECK-X64-LABEL:   0 | struct CX{{$}}
// CHECK-X64-NEXT:    0 |   (CX vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B5X (base)
// CHECK-X64-NEXT:   16 |     (B5X vbtable pointer)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:   28 |     int a1
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   48 |   struct A16X (virtual base) (empty)
// CHECK-X64-NEXT:   48 |   struct B2X (virtual base)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:   52 |   struct B6X (virtual base)
// CHECK-X64-NEXT:   52 |     int a
// CHECK-X64-NEXT:   76 |   (vtordisp for vbase B3X)
// CHECK-X64-NEXT:   80 |   struct B3X (virtual base)
// CHECK-X64-NEXT:   80 |     (B3X vftable pointer)
// CHECK-X64-NEXT:   88 |     int a
// CHECK-X64-NEXT:      | [sizeof=96, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct __declspec(align(16)) DX {
	int a;
	DX() : a(0xf000000D) {}
	virtual void f() {}
};

// CHECK-LABEL:   0 | struct DX{{$}}
// CHECK-NEXT:    0 |   (DX vftable pointer)
// CHECK-NEXT:    4 |   int a
// CHECK-NEXT:      | [sizeof=16, align=16
// CHECK-NEXT:      |  nvsize=8, nvalign=16]
// CHECK-X64-LABEL:   0 | struct DX{{$}}
// CHECK-X64-NEXT:    0 |   (DX vftable pointer)
// CHECK-X64-NEXT:    8 |   int a
// CHECK-X64-NEXT:      | [sizeof=16, align=16
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=16]

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
sizeof(AX)+
sizeof(BX)+
sizeof(CX)+
sizeof(DX)];
