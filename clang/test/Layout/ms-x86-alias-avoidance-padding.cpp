// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64 --strict-whitespace

extern "C" int printf(const char *fmt, ...);
__declspec(align(4096)) char buffer[4096];

struct AT {};

struct V : AT {
	char c;
	V() {
		printf("V   - this: %d\n", (int)((char*)this - buffer));
	}
};

struct AT0 {
	union { struct { int a; AT t; } y; int b; } x;
	char c;
	AT0() {
		printf("AT0 - this: %d\n", (int)((char*)this - buffer));
	}
};

struct AT1 : V {
	int a;
	AT1() {
		printf("AT1 - this: %d\n", (int)((char*)this - buffer));
	}
};

struct AT2 {
	AT0 t;
	char AT2FieldName0;
	AT2() {
		printf("AT2 - this: %d\n", (int)((char*)this - buffer));
		printf("AT2 - Fiel: %d\n", (int)((char*)&AT2FieldName0 - buffer));
	}
};

struct AT3 : AT2, AT1 {
	AT3() {
		printf("AT3 - this: %d\n", (int)((char*)this - buffer));
	}
};

// CHECK-LABEL:   0 | struct AT3{{$}}
// CHECK-NEXT:    0 |   struct AT2 (base)
// CHECK-NEXT:    0 |     struct AT0 t
// CHECK-NEXT:    0 |       union AT0::(unnamed at {{.*}} x
// CHECK-NEXT:    0 |         struct AT0::(unnamed at {{.*}} y
// CHECK-NEXT:    0 |           int a
// CHECK-NEXT:    4 |           struct AT t (empty)
// CHECK:         0 |         int b
// CHECK:         8 |       char c
// CHECK:        12 |     char AT2FieldName0
// CHECK-NEXT:   20 |   struct AT1 (base)
// CHECK-NEXT:   20 |     struct V (base)
// CHECK-NEXT:   20 |       struct AT (base) (empty)
// CHECK-NEXT:   20 |       char c
// CHECK-NEXT:   24 |     int a
// CHECK-NEXT:      | [sizeof=28, align=4
// CHECK-NEXT:      |  nvsize=28, nvalign=4]
// CHECK-X64-LABEL:   0 | struct AT3{{$}}
// CHECK-X64-NEXT:    0 |   struct AT2 (base)
// CHECK-X64-NEXT:    0 |     struct AT0 t
// CHECK-X64-NEXT:    0 |       union AT0::(unnamed at {{.*}} x
// CHECK-X64-NEXT:    0 |         struct AT0::(unnamed at {{.*}} y
// CHECK-X64-NEXT:    0 |           int a
// CHECK-X64-NEXT:    4 |           struct AT t (empty)
// CHECK-X64:         0 |         int b
// CHECK-X64:         8 |       char c
// CHECK-X64:        12 |     char AT2FieldName0
// CHECK-X64-NEXT:   20 |   struct AT1 (base)
// CHECK-X64-NEXT:   20 |     struct V (base)
// CHECK-X64-NEXT:   20 |       struct AT (base) (empty)
// CHECK-X64-NEXT:   20 |       char c
// CHECK-X64-NEXT:   24 |     int a
// CHECK-X64-NEXT:      | [sizeof=28, align=4
// CHECK-X64-NEXT:      |  nvsize=28, nvalign=4]

struct BT0 {
	BT0() {
		printf("BT0 - this: %d\n", (int)((char*)this - buffer));
	}
};

struct BT2 : BT0 {
	char BT2FieldName0;
	BT2() {
		printf("BT2 - this: %d\n", (int)((char*)this - buffer));
		printf("BT2 - Fiel: %d\n", (int)((char*)&BT2FieldName0 - buffer));
	}
};

struct BT3 : BT0, BT2 {
	BT3() {
		printf("BT3 - this: %d\n", (int)((char*)this - buffer));
	}
};

// CHECK-LABEL:   0 | struct BT3{{$}}
// CHECK-NEXT:    0 |   struct BT0 (base) (empty)
// CHECK-NEXT:    1 |   struct BT2 (base)
// CHECK-NEXT:    1 |     struct BT0 (base) (empty)
// CHECK-NEXT:    1 |     char BT2FieldName0
// CHECK-NEXT:      | [sizeof=2, align=1
// CHECK-NEXT:      |  nvsize=2, nvalign=1]
// CHECK-X64-LABEL:   0 | struct BT3{{$}}
// CHECK-X64-NEXT:    0 |   struct BT0 (base) (empty)
// CHECK-X64-NEXT:    1 |   struct BT2 (base)
// CHECK-X64-NEXT:    1 |     struct BT0 (base) (empty)
// CHECK-X64-NEXT:    1 |     char BT2FieldName0
// CHECK-X64-NEXT:      | [sizeof=2, align=1
// CHECK-X64-NEXT:      |  nvsize=2, nvalign=1]

struct T0 : AT {
	T0() {
		printf("T0 (this) : %d\n", (int)((char*)this - buffer));
	}
};

struct T1 : T0 {
	char a;
	T1() {
		printf("T1 (this) : %d\n", (int)((char*)this - buffer));
		printf("T1 (fiel) : %d\n", (int)((char*)&a - buffer));
	}
};

struct T2 : AT {
	char a;
	T2() {
		printf("T2 (this) : %d\n", (int)((char*)this - buffer));
		printf("T2 (fiel) : %d\n", (int)((char*)&a - buffer));
	}
};

struct __declspec(align(1)) T3 : virtual T1, virtual T2 {
	T3() {
		printf("T3 (this) : %d\n", (int)((char*)this - buffer));
	}
};

// CHECK-LABEL:   0 | struct T3{{$}}
// CHECK-NEXT:    0 |   (T3 vbtable pointer)
// CHECK-NEXT:    4 |   struct T1 (virtual base)
// CHECK-NEXT:    4 |     struct T0 (base) (empty)
// CHECK-NEXT:    4 |       struct AT (base) (empty)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:   12 |   struct T2 (virtual base)
// CHECK-NEXT:   12 |     struct AT (base) (empty)
// CHECK-NEXT:   12 |     char a
// CHECK-NEXT:      | [sizeof=16, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct T3{{$}}
// CHECK-X64-NEXT:    0 |   (T3 vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct T1 (virtual base)
// CHECK-X64-NEXT:    8 |     struct T0 (base) (empty)
// CHECK-X64-NEXT:    8 |       struct AT (base) (empty)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   16 |   struct T2 (virtual base)
// CHECK-X64-NEXT:   16 |     struct AT (base) (empty)
// CHECK-X64-NEXT:   16 |     char a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct B {};
struct C { int a; };
struct D : B, virtual C { B b; };
struct E : D, B {};
// CHECK-LABEL:   0 | struct E{{$}}
// CHECK-NEXT:    0 |   struct D (base)
// CHECK-NEXT:    4 |     struct B (base) (empty)
// CHECK-NEXT:    0 |     (D vbtable pointer)
// CHECK-NEXT:    4 |     struct B b (empty)
// CHECK:         8 |   struct B (base) (empty)
// CHECK-NEXT:    8 |   struct C (virtual base)
// CHECK-NEXT:    8 |     int a
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:  0 | struct E{{$}}
// CHECK-X64-NEXT:   0 |   struct D (base)
// CHECK-X64-NEXT:   8 |     struct B (base) (empty)
// CHECK-X64-NEXT:   0 |     (D vbtable pointer)
// CHECK-X64-NEXT:   8 |     struct B b (empty)
// CHECK-X64:       16 |   struct B (base) (empty)
// CHECK-X64-NEXT:  16 |   struct C (virtual base)
// CHECK-X64-NEXT:  16 |     int a
// CHECK-X64-NEXT:     | [sizeof=24, align=8
// CHECK-X64-NEXT:     |  nvsize=16, nvalign=8]

struct F : virtual D, virtual B {};
// CHECK-LABEL:   0 | struct F{{$}}
// CHECK-NEXT:    0 |   (F vbtable pointer)
// CHECK-NEXT:    4 |   struct C (virtual base)
// CHECK-NEXT:    4 |     int a
// CHECK-NEXT:    8 |   struct D (virtual base)
// CHECK-NEXT:   12 |     struct B (base) (empty)
// CHECK-NEXT:    8 |     (D vbtable pointer)
// CHECK-NEXT:   12 |     struct B b (empty)
// CHECK:        16 |   struct B (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=16, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct F{{$}}
// CHECK-X64-NEXT:    0 |   (F vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct C (virtual base)
// CHECK-X64-NEXT:    8 |     int a
// CHECK-X64-NEXT:   16 |   struct D (virtual base)
// CHECK-X64-NEXT:   24 |     struct B (base) (empty)
// CHECK-X64-NEXT:   16 |     (D vbtable pointer)
// CHECK-X64-NEXT:   24 |     struct B b (empty)
// CHECK-X64:        32 |   struct B (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=32, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct JC0 {
	JC0() { printf("JC0 : %d\n", (int)((char*)this - buffer)); }
};
struct JC1 : JC0 {
	virtual void f() {}
	JC1() { printf("JC1 : %d\n", (int)((char*)this - buffer)); }
};
struct JC2 : JC1 {
	JC2() { printf("JC2 : %d\n", (int)((char*)this - buffer)); }
};
struct JC4 : JC1, JC2 {
	JC4() { printf("JC4 : %d\n", (int)((char*)this - buffer)); }
};

// CHECK-LABEL:   0 | struct JC4{{$}}
// CHECK-NEXT:    0 |   struct JC1 (primary base)
// CHECK-NEXT:    0 |     (JC1 vftable pointer)
// CHECK-NEXT:    4 |     struct JC0 (base) (empty)
// CHECK-NEXT:    8 |   struct JC2 (base)
// CHECK-NEXT:    8 |     struct JC1 (primary base)
// CHECK-NEXT:    8 |       (JC1 vftable pointer)
// CHECK-NEXT:   12 |       struct JC0 (base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64-LABEL:   0 | struct JC4{{$}}
// CHECK-X64-NEXT:    0 |   struct JC1 (primary base)
// CHECK-X64-NEXT:    0 |     (JC1 vftable pointer)
// CHECK-X64-NEXT:    8 |     struct JC0 (base) (empty)
// CHECK-X64-NEXT:   16 |   struct JC2 (base)
// CHECK-X64-NEXT:   16 |     struct JC1 (primary base)
// CHECK-X64-NEXT:   16 |       (JC1 vftable pointer)
// CHECK-X64-NEXT:   24 |       struct JC0 (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct RA {};
struct RB { char c; };
struct RV {};
struct RW { char c; };
struct RY { RY() { printf("%Id\n", (char*)this - buffer); } };
struct RX0 : RB, RA {};
struct RX1 : RA, RB {};
struct RX2 : RA { char a; };
struct RX3 : RA { RB a; };
struct RX4 { RA a; char b; };
struct RX5 { RA a; RB b; };
struct RX6 : virtual RV { RB a; };
struct RX7 : virtual RW { RA a; };
struct RX8 : RA, virtual RW {};

struct RZ0 : RX0, RY {};
// CHECK-LABEL:   0 | struct RZ0{{$}}
// CHECK-NEXT:    0 |   struct RX0 (base)
// CHECK-NEXT:    0 |     struct RB (base)
// CHECK-NEXT:    0 |       char c
// CHECK-NEXT:    1 |     struct RA (base) (empty)
// CHECK-NEXT:    2 |   struct RY (base) (empty)
// CHECK-NEXT:      | [sizeof=2, align=1
// CHECK-NEXT:      |  nvsize=2, nvalign=1]
// CHECK-X64-LABEL:   0 | struct RZ0{{$}}
// CHECK-X64-NEXT:    0 |   struct RX0 (base)
// CHECK-X64-NEXT:    0 |     struct RB (base)
// CHECK-X64-NEXT:    0 |       char c
// CHECK-X64-NEXT:    1 |     struct RA (base) (empty)
// CHECK-X64-NEXT:    2 |   struct RY (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=2, align=1
// CHECK-X64-NEXT:      |  nvsize=2, nvalign=1]

struct RZ1 : RX1, RY {};
// CHECK-LABEL:   0 | struct RZ1{{$}}
// CHECK-NEXT:    0 |   struct RX1 (base)
// CHECK-NEXT:    0 |     struct RA (base) (empty)
// CHECK-NEXT:    0 |     struct RB (base)
// CHECK-NEXT:    0 |       char c
// CHECK-NEXT:    1 |   struct RY (base) (empty)
// CHECK-NEXT:      | [sizeof=1, align=1
// CHECK-NEXT:      |  nvsize=1, nvalign=1]
// CHECK-X64-LABEL:   0 | struct RZ1{{$}}
// CHECK-X64-NEXT:    0 |   struct RX1 (base)
// CHECK-X64-NEXT:    0 |     struct RA (base) (empty)
// CHECK-X64-NEXT:    0 |     struct RB (base)
// CHECK-X64-NEXT:    0 |       char c
// CHECK-X64-NEXT:    1 |   struct RY (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=1, align=1
// CHECK-X64-NEXT:      |  nvsize=1, nvalign=1]

struct RZ2 : RX2, RY {};
// CHECK-LABEL:   0 | struct RZ2{{$}}
// CHECK-NEXT:    0 |   struct RX2 (base)
// CHECK-NEXT:    0 |     struct RA (base) (empty)
// CHECK-NEXT:    0 |     char a
// CHECK-NEXT:    2 |   struct RY (base) (empty)
// CHECK-NEXT:      | [sizeof=2, align=1
// CHECK-NEXT:      |  nvsize=2, nvalign=1]
// CHECK-X64-LABEL:   0 | struct RZ2{{$}}
// CHECK-X64-NEXT:    0 |   struct RX2 (base)
// CHECK-X64-NEXT:    0 |     struct RA (base) (empty)
// CHECK-X64-NEXT:    0 |     char a
// CHECK-X64-NEXT:    2 |   struct RY (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=2, align=1
// CHECK-X64-NEXT:      |  nvsize=2, nvalign=1]

struct RZ3 : RX3, RY {};
// CHECK-LABEL:   0 | struct RZ3{{$}}
// CHECK-NEXT:    0 |   struct RX3 (base)
// CHECK-NEXT:    0 |     struct RA (base) (empty)
// CHECK-NEXT:    0 |     struct RB a
// CHECK-NEXT:    0 |       char c
// CHECK-NEXT:    1 |   struct RY (base) (empty)
// CHECK-NEXT:      | [sizeof=1, align=1
// CHECK-NEXT:      |  nvsize=1, nvalign=1]
// CHECK-X64-LABEL:   0 | struct RZ3{{$}}
// CHECK-X64-NEXT:    0 |   struct RX3 (base)
// CHECK-X64-NEXT:    0 |     struct RA (base) (empty)
// CHECK-X64-NEXT:    0 |     struct RB a
// CHECK-X64-NEXT:    0 |       char c
// CHECK-X64-NEXT:    1 |   struct RY (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=1, align=1
// CHECK-X64-NEXT:      |  nvsize=1, nvalign=1]

struct RZ4 : RX4, RY {};
// CHECK-LABEL:   0 | struct RZ4{{$}}
// CHECK-NEXT:    0 |   struct RX4 (base)
// CHECK-NEXT:    0 |     struct RA a (empty)
// CHECK-NEXT:    1 |     char b
// CHECK-NEXT:    3 |   struct RY (base) (empty)
// CHECK-NEXT:      | [sizeof=3, align=1
// CHECK-NEXT:      |  nvsize=3, nvalign=1]
// CHECK-X64-LABEL:   0 | struct RZ4{{$}}
// CHECK-X64-NEXT:    0 |   struct RX4 (base)
// CHECK-X64-NEXT:    0 |     struct RA a (empty)
// CHECK-X64-NEXT:    1 |     char b
// CHECK-X64-NEXT:    3 |   struct RY (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=3, align=1
// CHECK-X64-NEXT:      |  nvsize=3, nvalign=1]

struct RZ5 : RX5, RY {};
// CHECK-LABEL:   0 | struct RZ5{{$}}
// CHECK-NEXT:    0 |   struct RX5 (base)
// CHECK-NEXT:    0 |     struct RA a (empty)
// CHECK-NEXT:    1 |     struct RB b
// CHECK-NEXT:    1 |       char c
// CHECK-NEXT:    2 |   struct RY (base) (empty)
// CHECK-NEXT:      | [sizeof=2, align=1
// CHECK-NEXT:      |  nvsize=2, nvalign=1]
// CHECK-X64-LABEL:   0 | struct RZ5{{$}}
// CHECK-X64-NEXT:    0 |   struct RX5 (base)
// CHECK-X64-NEXT:    0 |     struct RA a (empty)
// CHECK-X64-NEXT:    1 |     struct RB b
// CHECK-X64-NEXT:    1 |       char c
// CHECK-X64-NEXT:    2 |   struct RY (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=2, align=1
// CHECK-X64-NEXT:      |  nvsize=2, nvalign=1]

struct RZ6 : RX6, RY {};
// CHECK-LABEL:   0 | struct RZ6{{$}}
// CHECK-NEXT:    0 |   struct RX6 (base)
// CHECK-NEXT:    0 |     (RX6 vbtable pointer)
// CHECK-NEXT:    4 |     struct RB a
// CHECK-NEXT:    4 |       char c
// CHECK-NEXT:    9 |   struct RY (base) (empty)
// CHECK-NEXT:   12 |   struct RV (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=12, nvalign=4]
// CHECK-X64-LABEL:   0 | struct RZ6{{$}}
// CHECK-X64-NEXT:    0 |   struct RX6 (base)
// CHECK-X64-NEXT:    0 |     (RX6 vbtable pointer)
// CHECK-X64-NEXT:    8 |     struct RB a
// CHECK-X64-NEXT:    8 |       char c
// CHECK-X64-NEXT:   17 |   struct RY (base) (empty)
// CHECK-X64-NEXT:   24 |   struct RV (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

struct RZ7 : RX7, RY {};
// CHECK-LABEL:   0 | struct RZ7{{$}}
// CHECK-NEXT:    0 |   struct RX7 (base)
// CHECK-NEXT:    0 |     (RX7 vbtable pointer)
// CHECK-NEXT:    4 |     struct RA a (empty)
// CHECK-NEXT:    8 |   struct RY (base) (empty)
// CHECK-NEXT:    8 |   struct RW (virtual base)
// CHECK-NEXT:    8 |     char c
// CHECK-NEXT:      | [sizeof=9, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct RZ7{{$}}
// CHECK-X64-NEXT:    0 |   struct RX7 (base)
// CHECK-X64-NEXT:    0 |     (RX7 vbtable pointer)
// CHECK-X64-NEXT:    8 |     struct RA a (empty)
// CHECK-X64-NEXT:   16 |   struct RY (base) (empty)
// CHECK-X64-NEXT:   16 |   struct RW (virtual base)
// CHECK-X64-NEXT:   16 |     char c
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=8]

struct RZ8 : RX8, RY {};
// CHECK-LABEL:   0 | struct RZ8{{$}}
// CHECK-NEXT:    0 |   struct RX8 (base)
// CHECK-NEXT:    4 |     struct RA (base) (empty)
// CHECK-NEXT:    0 |     (RX8 vbtable pointer)
// CHECK-NEXT:    4 |   struct RY (base) (empty)
// CHECK-NEXT:    4 |   struct RW (virtual base)
// CHECK-NEXT:    4 |     char c
// CHECK-NEXT:      | [sizeof=5, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct RZ8{{$}}
// CHECK-X64-NEXT:    0 |   struct RX8 (base)
// CHECK-X64-NEXT:    8 |     struct RA (base) (empty)
// CHECK-X64-NEXT:    0 |     (RX8 vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct RY (base) (empty)
// CHECK-X64-NEXT:    8 |   struct RW (virtual base)
// CHECK-X64-NEXT:    8 |     char c
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

struct JA {};
struct JB {};
struct JC : JA { virtual void f() {} };
struct JD : virtual JB, virtual JC { virtual void f() {} JD() {} };

// CHECK-LABEL:   0 | struct JD{{$}}
// CHECK-NEXT:    0 |   (JD vbtable pointer)
// CHECK-NEXT:    4 |   struct JB (virtual base) (empty)
// CHECK-NEXT:    4 |   (vtordisp for vbase JC)
// CHECK-NEXT:    8 |   struct JC (virtual base)
// CHECK-NEXT:    8 |     (JC vftable pointer)
// CHECK-NEXT:   12 |     struct JA (base) (empty)
// CHECK-NEXT:      | [sizeof=12, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct JD{{$}}
// CHECK-X64-NEXT:    0 |   (JD vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct JB (virtual base) (empty)
// CHECK-X64-NEXT:   12 |   (vtordisp for vbase JC)
// CHECK-X64-NEXT:   16 |   struct JC (virtual base)
// CHECK-X64-NEXT:   16 |     (JC vftable pointer)
// CHECK-X64-NEXT:   24 |     struct JA (base) (empty)
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]

int a[
sizeof(AT3) +
sizeof(BT3) +
sizeof(T3) +
sizeof(E) +
sizeof(F) +
sizeof(JC4) +
sizeof(RZ0) +
sizeof(RZ1) +
sizeof(RZ2) +
sizeof(RZ3) +
sizeof(RZ4) +
sizeof(RZ5) +
sizeof(RZ6) +
sizeof(RZ7) +
sizeof(RZ8) +
sizeof(JD) +
0];
