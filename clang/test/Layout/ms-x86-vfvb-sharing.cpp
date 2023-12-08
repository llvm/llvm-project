// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>&1 \
// RUN:            | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64 --strict-whitespace

extern "C" int printf(const char *fmt, ...);

struct B0 { int a; B0() : a(0xf00000B0) { printf("B0 = %p\n", this); } };
struct B1 { int a; B1() : a(0xf00000B1) { printf("B1 = %p\n", this); } };
struct B2 { B2() { printf("B2 = %p\n", this); } virtual void g() { printf("B2"); } };
struct B3 : virtual B1 { B3() { printf("B3 = %p\n", this); } };
struct B4 : virtual B1 { B4() { printf("B4 = %p\n", this); } virtual void g() { printf("B4"); } };

struct A : B0, virtual B1 {
	__declspec(align(16)) int a;
	A() : a(0xf000000A) { printf(" A = %p\n\n", this); }
	virtual void f() { printf("A"); }
};

// CHECK-LABEL:   0 | struct A
// CHECK-NEXT:    0 |   (A vftable pointer)
// CHECK-NEXT:   16 |   struct B0 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |   (A vbtable pointer)
// CHECK-NEXT:   48 |   int a
// CHECK-NEXT:   64 |   struct B1 (virtual base)
// CHECK-NEXT:   64 |     int a
// CHECK-NEXT:      | [sizeof=80, align=16
// CHECK-NEXT:      |  nvsize=64, nvalign=16]
// CHECK-X64-LABEL:   0 | struct A
// CHECK-X64-NEXT:    0 |   (A vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B0 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   24 |   (A vbtable pointer)
// CHECK-X64-NEXT:   48 |   int a
// CHECK-X64-NEXT:   64 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   64 |     int a
// CHECK-X64-NEXT:      | [sizeof=80, align=16
// CHECK-X64-NEXT:      |  nvsize=64, nvalign=16]

struct B : B2, B0, virtual B1 {
	__declspec(align(16)) int a;
	B() : a(0xf000000B) { printf(" B = %p\n\n", this); }
	virtual void f() { printf("B"); }
};

// CHECK-LABEL:   0 | struct B{{$}}
// CHECK-NEXT:    0 |   struct B2 (primary base)
// CHECK-NEXT:    0 |     (B2 vftable pointer)
// CHECK-NEXT:    4 |   struct B0 (base)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   (B vbtable pointer)
// CHECK-NEXT:   32 |   int a
// CHECK-NEXT:   48 |   struct B1 (virtual base)
// CHECK-NEXT:   48 |     int a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct B{{$}}
// CHECK-X64-NEXT:    0 |   struct B2 (primary base)
// CHECK-X64-NEXT:    0 |     (B2 vftable pointer)
// CHECK-X64-NEXT:    8 |   struct B0 (base)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   (B vbtable pointer)
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   48 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct C : B3, B0, virtual B1 {
	__declspec(align(16)) int a;
	C() : a(0xf000000C) { printf(" C = %p\n\n", this); }
	virtual void f() { printf("C"); }
};

// CHECK-LABEL:   0 | struct C
// CHECK-NEXT:    0 |   (C vftable pointer)
// CHECK-NEXT:   16 |   struct B3 (base)
// CHECK-NEXT:   16 |     (B3 vbtable pointer)
// CHECK-NEXT:   20 |   struct B0 (base)
// CHECK-NEXT:   20 |     int a
// CHECK-NEXT:   32 |   int a
// CHECK-NEXT:   48 |   struct B1 (virtual base)
// CHECK-NEXT:   48 |     int a
// CHECK-NEXT:      | [sizeof=64, align=16
// CHECK-NEXT:      |  nvsize=48, nvalign=16]
// CHECK-X64-LABEL:   0 | struct C
// CHECK-X64-NEXT:    0 |   (C vftable pointer)
// CHECK-X64-NEXT:   16 |   struct B3 (base)
// CHECK-X64-NEXT:   16 |     (B3 vbtable pointer)
// CHECK-X64-NEXT:   24 |   struct B0 (base)
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   48 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

struct D : B4, B0, virtual B1 {
	__declspec(align(16)) int a;
	D() : a(0xf000000D) { printf(" D = %p\n\n", this); }
	virtual void f() { printf("D"); }
};

// CHECK-LABEL:   0 | struct D
// CHECK-NEXT:    0 |   struct B4 (primary base)
// CHECK-NEXT:    0 |     (B4 vftable pointer)
// CHECK-NEXT:    4 |     (B4 vbtable pointer)
// CHECK-NEXT:    8 |   struct B0 (base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:   16 |   int a
// CHECK-NEXT:   32 |   struct B1 (virtual base)
// CHECK-NEXT:   32 |     int a
// CHECK-NEXT:      | [sizeof=48, align=16
// CHECK-NEXT:      |  nvsize=32, nvalign=16]
// CHECK-X64-LABEL:   0 | struct D
// CHECK-X64-NEXT:    0 |   struct B4 (primary base)
// CHECK-X64-NEXT:    0 |     (B4 vftable pointer)
// CHECK-X64-NEXT:    8 |     (B4 vbtable pointer)
// CHECK-X64-NEXT:   16 |   struct B0 (base)
// CHECK-X64-NEXT:   16 |     int a
// CHECK-X64-NEXT:   32 |   int a
// CHECK-X64-NEXT:   48 |   struct B1 (virtual base)
// CHECK-X64-NEXT:   48 |     int a
// CHECK-X64-NEXT:      | [sizeof=64, align=16
// CHECK-X64-NEXT:      |  nvsize=48, nvalign=16]

int a[
sizeof(A)+
sizeof(B)+
sizeof(C)+
sizeof(D)];
