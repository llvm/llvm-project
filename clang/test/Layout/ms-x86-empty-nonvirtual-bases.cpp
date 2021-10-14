// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s --strict-whitespace

extern "C" int printf(const char *fmt, ...);

struct __declspec(align(8)) B0 { B0() {printf("B0 : %p\n", this);} };
struct __declspec(align(8)) B1 { B1() {printf("B1 : %p\n", this);} };
struct __declspec(align(8)) B2 { B2() {printf("B2 : %p\n", this);} };
struct __declspec(align(8)) B3 { B3() {printf("B3 : %p\n", this);} };
struct __declspec(align(8)) B4 { B4() {printf("B4 : %p\n", this);} };

struct C0 { int a; C0() : a(0xf00000C0) {printf("C0 : %p\n", this);} };
struct C1 { int a; C1() : a(0xf00000C1) {printf("C1 : %p\n", this);} };
struct C2 { int a; C2() : a(0xf00000C2) {printf("C2 : %p\n", this);} };
struct C3 { int a; C3() : a(0xf00000C3) {printf("C3 : %p\n", this);} };
struct C4 { int a; C4() : a(0xf00000C4) {printf("C4 : %p\n", this);} };

struct A : B0 {
	int a;
	A() : a(0xf000000A) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct A{{$}}
// CHECK-NEXT:    0 |   struct B0 (base) (empty)
// CHECK-NEXT:    0 |   int a
// CHECK-NEXT:      | [sizeof=8, align=8
// CHECK-NEXT:      |  nvsize=8, nvalign=8]

struct B : B0 {
	B0 b0;
	int a;
	B() : a(0xf000000B) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct B{{$}}
// CHECK-NEXT:    0 |   struct B0 (base) (empty)
// CHECK-NEXT:    0 |   struct B0 b0 (empty)
// CHECK:         8 |   int a
// CHECK-NEXT:      | [sizeof=16, align=8
// CHECK-NEXT:      |  nvsize=16, nvalign=8]

struct C : B0, B1, B2, B3, B4 {
	int a;
	C() : a(0xf000000C) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct C{{$}}
// CHECK-NEXT:    0 |   struct B0 (base) (empty)
// CHECK-NEXT:    8 |   struct B1 (base) (empty)
// CHECK-NEXT:   16 |   struct B2 (base) (empty)
// CHECK-NEXT:   24 |   struct B3 (base) (empty)
// CHECK-NEXT:   32 |   struct B4 (base) (empty)
// CHECK-NEXT:   32 |   int a
// CHECK-NEXT:      | [sizeof=40, align=8
// CHECK-NEXT:      |  nvsize=40, nvalign=8]

struct D {
	B0 b0;
	C0 c0;
	C1 c1;
	C2 c2;
	B1 b1;
	int a;
	D() : a(0xf000000D) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct D{{$}}
// CHECK-NEXT:    0 |   struct B0 b0 (empty)
// CHECK:         8 |   struct C0 c0
// CHECK-NEXT:    8 |     int a
// CHECK:        12 |   struct C1 c1
// CHECK-NEXT:   12 |     int a
// CHECK:        16 |   struct C2 c2
// CHECK-NEXT:   16 |     int a
// CHECK:        24 |   struct B1 b1 (empty)
// CHECK:        32 |   int a
// CHECK-NEXT:      | [sizeof=40, align=8
// CHECK-NEXT:      |  nvsize=40, nvalign=8]

struct E : B0, C0, C1, C2, B1 {
	int a;
	E() : a(0xf000000E) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct E{{$}}
// CHECK-NEXT:    0 |   struct B0 (base) (empty)
// CHECK-NEXT:    0 |   struct C0 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:    4 |   struct C1 (base)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   struct C2 (base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:   16 |   struct B1 (base) (empty)
// CHECK-NEXT:   16 |   int a
// CHECK-NEXT:      | [sizeof=24, align=8
// CHECK-NEXT:      |  nvsize=24, nvalign=8]

struct F : C0, B0, B1, C1 {
	int a;
	F() : a(0xf000000F) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct F{{$}}
// CHECK-NEXT:    0 |   struct C0 (base)
// CHECK-NEXT:    0 |     int a
// CHECK-NEXT:    8 |   struct B0 (base) (empty)
// CHECK-NEXT:   16 |   struct B1 (base) (empty)
// CHECK-NEXT:   16 |   struct C1 (base)
// CHECK-NEXT:   16 |     int a
// CHECK-NEXT:   20 |   int a
// CHECK-NEXT:      | [sizeof=24, align=8
// CHECK-NEXT:      |  nvsize=24, nvalign=8]

struct G : B0, B1, B2, B3, B4 {
	__declspec(align(32)) int a;
	G() : a(0xf0000011) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct G{{$}}
// CHECK-NEXT:    0 |   struct B0 (base) (empty)
// CHECK-NEXT:    8 |   struct B1 (base) (empty)
// CHECK-NEXT:   16 |   struct B2 (base) (empty)
// CHECK-NEXT:   24 |   struct B3 (base) (empty)
// CHECK-NEXT:   32 |   struct B4 (base) (empty)
// CHECK-NEXT:   32 |   int a
// CHECK-NEXT:      | [sizeof=64, align=32
// CHECK-NEXT:      |  nvsize=64, nvalign=32]

struct __declspec(align(32)) H : B0, B1, B2, B3, B4 {
	int a;
	H() : a(0xf0000011) {printf("X : %p\n", this);}
};

// CHECK-LABEL:   0 | struct H{{$}}
// CHECK-NEXT:    0 |   struct B0 (base) (empty)
// CHECK-NEXT:    8 |   struct B1 (base) (empty)
// CHECK-NEXT:   16 |   struct B2 (base) (empty)
// CHECK-NEXT:   24 |   struct B3 (base) (empty)
// CHECK-NEXT:   32 |   struct B4 (base) (empty)
// CHECK-NEXT:   32 |   int a
// CHECK-NEXT:      | [sizeof=64, align=32
// CHECK-NEXT:      |  nvsize=40, nvalign=32]

struct I {
  int i0[0];
};

// CHECK-LABEL:   0 | struct I{{$}}
// CHECK-NEXT:    0 |   int [0] i0
// CHECK-NEXT:      | [sizeof={{1|4}}, align=4,
// CHECK-NEXT:      |  nvsize=0, nvalign=4]

struct J : I {
  int j;
};

// CHECK-LABEL:  0 | struct J{{$}}
// CHECK-NEXT:   0 |   struct I (base)
// CHECK-NEXT:   0 |     int [0] i0
// CHECK-NEXT:   0 |   int j
// CHECK-NEXT:     | [sizeof=4, align=4,
// CHECK-NEXT:     |  nvsize=4, nvalign=4]

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
sizeof(J)];
