// RUN: %clang_cc1 -fno-rtti -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only -Wno-inaccessible-base %s 2>&1 \
// RUN:            | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -fno-rtti -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only -Wno-inaccessible-base %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64 --strict-whitespace

extern "C" int printf(const char *fmt, ...);
char buffer[419430400];

struct A {
	char a;
	A() {
		printf("A   = %d\n", (int)((char*)this - buffer));
		printf("A.a = %d\n", (int)((char*)&a - buffer));
	}
};

struct B {
	__declspec(align(4)) long long a;
	B() {
		printf("B   = %d\n", (int)((char*)this - buffer));
		printf("B.a = %d\n", (int)((char*)&a - buffer));
	}
};

#pragma pack(push, 2)
struct X {
	B a;
	char b;
	int c;
	X() {
		printf("X   = %d\n", (int)((char*)this - buffer));
		printf("X.a = %d\n", (int)((char*)&a - buffer));
		printf("X.b = %d\n", (int)((char*)&b - buffer));
		printf("X.c = %d\n", (int)((char*)&c - buffer));
	}
};

// CHECK-LABEL:   0 | struct X{{$}}
// CHECK-NEXT:    0 |   struct B a
// CHECK-NEXT:    0 |     long long a
// CHECK-NEXT:    8 |   char b
// CHECK-NEXT:   10 |   int c
// CHECK-NEXT:      | [sizeof=16, align=4
// CHECK-NEXT:      |  nvsize=14, nvalign=4]
// CHECK-X64-LABEL:   0 | struct X{{$}}
// CHECK-X64-NEXT:    0 |   struct B a
// CHECK-X64-NEXT:    0 |     long long a
// CHECK-X64-NEXT:    8 |   char b
// CHECK-X64-NEXT:   10 |   int c
// CHECK-X64-NEXT:      | [sizeof=16, align=4
// CHECK-X64-NEXT:      |  nvsize=14, nvalign=4]

struct Y : A, B {
	char a;
	int b;
	Y() {
		printf("Y   = %d\n", (int)((char*)this - buffer));
		printf("Y.a = %d\n", (int)((char*)&a - buffer));
		printf("Y.b = %d\n", (int)((char*)&b - buffer));
	}
};

// CHECK-LABEL:   0 | struct Y{{$}}
// CHECK-NEXT:    0 |   struct A (base)
// CHECK-NEXT:    0 |     char a
// CHECK-NEXT:    4 |   struct B (base)
// CHECK-NEXT:    4 |     long long a
// CHECK-NEXT:   12 |   char a
// CHECK-NEXT:   14 |   int b
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=18, nvalign=4]
// CHECK-X64-LABEL:   0 | struct Y{{$}}
// CHECK-X64-NEXT:    0 |   struct A (base)
// CHECK-X64-NEXT:    0 |     char a
// CHECK-X64-NEXT:    4 |   struct B (base)
// CHECK-X64-NEXT:    4 |     long long a
// CHECK-X64-NEXT:   12 |   char a
// CHECK-X64-NEXT:   14 |   int b
// CHECK-X64-NEXT:      | [sizeof=20, align=4
// CHECK-X64-NEXT:      |  nvsize=18, nvalign=4]

struct Z : virtual B {
	char a;
	int b;
	Z() {
		printf("Z   = %d\n", (int)((char*)this - buffer));
		printf("Z.a = %d\n", (int)((char*)&a - buffer));
		printf("Z.b = %d\n", (int)((char*)&b - buffer));
	}
};

// CHECK-LABEL:   0 | struct Z{{$}}
// CHECK-NEXT:    0 |   (Z vbtable pointer)
// CHECK-NEXT:    4 |   char a
// CHECK-NEXT:    6 |   int b
// CHECK-NEXT:   12 |   struct B (virtual base)
// CHECK-NEXT:   12 |     long long a
// CHECK-NEXT:      | [sizeof=20, align=4
// CHECK-NEXT:      |  nvsize=10, nvalign=4]
// CHECK-X64-LABEL:   0 | struct Z{{$}}
// CHECK-X64-NEXT:    0 |   (Z vbtable pointer)
// CHECK-X64-NEXT:    8 |   char a
// CHECK-X64-NEXT:   10 |   int b
// CHECK-X64-NEXT:   16 |   struct B (virtual base)
// CHECK-X64-NEXT:   16 |     long long a
// CHECK-X64-NEXT:      | [sizeof=24, align=4
// CHECK-X64-NEXT:      |  nvsize=14, nvalign=4]

#pragma pack(pop)

struct A1 { long long a; };
#pragma pack(push, 1)
struct B1 : virtual A1 { char a; };
#pragma pack(pop)
struct C1 : B1 {};

// CHECK-LABEL:   0 | struct C1{{$}}
// CHECK-NEXT:    0 |   struct B1 (base)
// CHECK-NEXT:    0 |     (B1 vbtable pointer)
// CHECK-NEXT:    4 |     char a
// CHECK-NEXT:    8 |   struct A1 (virtual base)
// CHECK-NEXT:    8 |     long long a
// CHECK-NEXT:      | [sizeof=16, align=8
// CHECK-NEXT:      |  nvsize=5, nvalign=8]
// CHECK-X64-LABEL:   0 | struct C1{{$}}
// CHECK-X64-NEXT:    0 |   struct B1 (base)
// CHECK-X64-NEXT:    0 |     (B1 vbtable pointer)
// CHECK-X64-NEXT:    8 |     char a
// CHECK-X64-NEXT:   16 |   struct A1 (virtual base)
// CHECK-X64-NEXT:   16 |     long long a
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=9, nvalign=8]

struct CA0 {
	CA0() {}
};
struct CA1 : virtual CA0 {
	CA1() {}
};
#pragma pack(push, 1)
struct CA2 : public CA1, public CA0 {
	virtual void CA2Method() {}
	CA2() {}
};
#pragma pack(pop)

// CHECK-LABEL:   0 | struct CA2{{$}}
// CHECK-NEXT:    0 |   (CA2 vftable pointer)
// CHECK-NEXT:    4 |   struct CA1 (base)
// CHECK-NEXT:    4 |     (CA1 vbtable pointer)
// CHECK-NEXT:    9 |   struct CA0 (base) (empty)
// CHECK-NEXT:    9 |   struct CA0 (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=9, align=1
// CHECK-NEXT:      |  nvsize=9, nvalign=1]
// CHECK-X64-LABEL:   0 | struct CA2{{$}}
// CHECK-X64-NEXT:    0 |   (CA2 vftable pointer)
// CHECK-X64-NEXT:    8 |   struct CA1 (base)
// CHECK-X64-NEXT:    8 |     (CA1 vbtable pointer)
// CHECK-X64-NEXT:   17 |   struct CA0 (base) (empty)
// CHECK-X64-NEXT:   17 |   struct CA0 (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=17, align=1
// CHECK-X64-NEXT:      |  nvsize=17, nvalign=1]

#pragma pack(16)
struct YA {
	__declspec(align(32)) char : 1;
};
// CHECK-LABEL:   0 | struct YA{{$}}
// CHECK-NEXT:0:0-0 |   char
// CHECK-NEXT:      | [sizeof=32, align=32
// CHECK-NEXT:      |  nvsize=32, nvalign=32]
// CHECK-X64-LABEL:   0 | struct YA{{$}}
// CHECK-X64-NEXT:0:0-0 |   char
// CHECK-X64-NEXT:      | [sizeof=32, align=32
// CHECK-X64-NEXT:      |  nvsize=32, nvalign=32]

#pragma pack(1)
struct YB {
	char a;
	YA b;
};
// CHECK-LABEL:   0 | struct YB{{$}}
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    1 |   struct YA b
// CHECK-NEXT:1:0-0 |     char
// CHECK-NEXT:      | [sizeof=33, align=1
// CHECK-NEXT:      |  nvsize=33, nvalign=1]
// CHECK-X64-LABEL:   0 | struct YB{{$}}
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:    1 |   struct YA b
// CHECK-X64-NEXT:1:0-0 |     char
// CHECK-X64-NEXT:      | [sizeof=33, align=1
// CHECK-X64-NEXT:      |  nvsize=33, nvalign=1]

#pragma pack(8)
struct YC {
	__declspec(align(32)) char : 1;
};
// CHECK-LABEL:   0 | struct YC{{$}}
// CHECK-NEXT:0:0-0 |   char
// CHECK-NEXT:      | [sizeof=32, align=32
// CHECK-NEXT:      |  nvsize=32, nvalign=32]
// CHECK-X64-LABEL:   0 | struct YC{{$}}
// CHECK-X64-NEXT:    0:0-0 |   char
// CHECK-X64-NEXT:      | [sizeof=8, align=32
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=32]

#pragma pack(1)
struct YD {
	char a;
	YC b;
};
// CHECK-LABEL:   0 | struct YD{{$}}
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    1 |   struct YC b
// CHECK-NEXT:1:0-0 |     char
// CHECK-NEXT:      | [sizeof=33, align=1
// CHECK-NEXT:      |  nvsize=33, nvalign=1]
// CHECK-X64-LABEL:   0 | struct YD{{$}}
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:    1 |   struct YC b
// CHECK-X64-NEXT:1:0-0 |     char
// CHECK-X64-NEXT:      | [sizeof=9, align=1
// CHECK-X64-NEXT:      |  nvsize=9, nvalign=1]

#pragma pack(4)
struct YE {
	__declspec(align(32)) char : 1;
};
// CHECK-LABEL:   0 | struct YE{{$}}
// CHECK-NEXT:    0:0-0 |   char
// CHECK-NEXT:      | [sizeof=4, align=32
// CHECK-NEXT:      |  nvsize=4, nvalign=32]
// CHECK-X64-LABEL:   0 | struct YE{{$}}
// CHECK-X64-NEXT:    0:0-0 |   char
// CHECK-X64-NEXT:      | [sizeof=4, align=32
// CHECK-X64-NEXT:      |  nvsize=4, nvalign=32]

#pragma pack(1)
struct YF {
	char a;
	YE b;
};
// CHECK-LABEL:   0 | struct YF{{$}}
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    1 |   struct YE b
// CHECK-NEXT:1:0-0 |     char
// CHECK-NEXT:      | [sizeof=5, align=1
// CHECK-NEXT:      |  nvsize=5, nvalign=1]
// CHECK-X64-LABEL:   0 | struct YF{{$}}
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:    1 |   struct YE b
// CHECK-X64-NEXT:1:0-0 |     char
// CHECK-X64-NEXT:      | [sizeof=5, align=1
// CHECK-X64-NEXT:      |  nvsize=5, nvalign=1]

#pragma pack(16)
struct __declspec(align(16)) D0 { char a; };
#pragma pack(1)
struct D1 : public D0 { char a; };
#pragma pack(16)
struct D2 : D1 { char a; };

// CHECK-LABEL:   0 | struct D2{{$}}
// CHECK-NEXT:    0 |   struct D1 (base)
// CHECK-NEXT:    0 |     struct D0 (base)
// CHECK-NEXT:    0 |       char a
// CHECK-NEXT:    1 |     char a
// CHECK-NEXT:    2 |   char a
// CHECK-NEXT:      | [sizeof=16, align=16
// CHECK-NEXT:      |  nvsize=16, nvalign=16]
// CHECK-X64-LABEL:   0 | struct D2{{$}}
// CHECK-X64-NEXT:    0 |   struct D1 (base)
// CHECK-X64-NEXT:    0 |     struct D0 (base)
// CHECK-X64-NEXT:    0 |       char a
// CHECK-X64-NEXT:    1 |     char a
// CHECK-X64-NEXT:    2 |   char a
// CHECK-X64-NEXT:      | [sizeof=16, align=16
// CHECK-X64-NEXT:      |  nvsize=16, nvalign=16]

#pragma pack()
struct JA { char a; };
#pragma pack(1)
struct JB { __declspec(align(4)) char a; };
#pragma pack()
struct JC : JB, JA { };

// CHECK-LABEL:   0 | struct JC{{$}}
// CHECK-NEXT:    0 |   struct JB (base)
// CHECK-NEXT:    0 |     char a
// CHECK-NEXT:    1 |   struct JA (base)
// CHECK-NEXT:    1 |     char a
// CHECK-NEXT:      | [sizeof=4, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct JC{{$}}
// CHECK-X64-NEXT:    0 |   struct JB (base)
// CHECK-X64-NEXT:    0 |     char a
// CHECK-X64-NEXT:    1 |   struct JA (base)
// CHECK-X64-NEXT:    1 |     char a
// CHECK-X64-NEXT:      | [sizeof=4, align=4
// CHECK-X64-NEXT:      |  nvsize=4, nvalign=4]

#pragma pack()
struct KA { char a; };
#pragma pack(1)
struct KB : KA { __declspec(align(2)) char a; };

// CHECK-LABEL:   0 | struct KB{{$}}
// CHECK-NEXT:    0 |   struct KA (base)
// CHECK-NEXT:    0 |     char a
// CHECK-NEXT:    2 |   char a
// CHECK-NEXT:      | [sizeof=4, align=2
// CHECK-NEXT:      |  nvsize=3, nvalign=2]
// CHECK-X64-LABEL:   0 | struct KB{{$}}
// CHECK-X64-NEXT:    0 |   struct KA (base)
// CHECK-X64-NEXT:    0 |     char a
// CHECK-X64-NEXT:    2 |   char a
// CHECK-X64-NEXT:      | [sizeof=4, align=2
// CHECK-X64-NEXT:      |  nvsize=3, nvalign=2]

#pragma pack(1)
struct L {
  virtual void fun() {}
  __declspec(align(256)) int Field;
};

// CHECK-LABEL:   0 | struct L{{$}}
// CHECK-NEXT:    0 |   (L vftable pointer)
// CHECK-NEXT:  256 |   int Field
// CHECK-NEXT:      | [sizeof=512, align=256
// CHECK-NEXT:      |  nvsize=260, nvalign=256]
// CHECK-X64-LABEL:   0 | struct L{{$}}
// CHECK-X64-NEXT:    0 |   (L vftable pointer)
// CHECK-X64-NEXT:  256 |   int Field
// CHECK-X64-NEXT:      | [sizeof=512, align=256
// CHECK-X64-NEXT:      |  nvsize=260, nvalign=256]

#pragma pack()
struct MA {};
#pragma pack(1)
struct MB : virtual MA {
  __declspec(align(256)) int Field;
};

// CHECK-LABEL:   0 | struct MB{{$}}
// CHECK-NEXT:    0 |   (MB vbtable pointer)
// CHECK-NEXT:  256 |   int Field
// CHECK-NEXT:  260 |   struct MA (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=512, align=256
// CHECK-NEXT:      |  nvsize=260, nvalign=256]
// CHECK-X64-LABEL:   0 | struct MB{{$}}
// CHECK-X64-NEXT:    0 |   (MB vbtable pointer)
// CHECK-X64-NEXT:  256 |   int Field
// CHECK-X64-NEXT:  260 |   struct MA (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=512, align=256
// CHECK-X64-NEXT:      |  nvsize=260, nvalign=256]

struct RA {};
#pragma pack(1)
struct __declspec(align(8)) RB0 { 
	__declspec(align(1024)) int b : 3;
};

struct __declspec(align(8)) RB1 { 
	__declspec(align(1024)) int b : 3;
	virtual void f() {}
};

struct __declspec(align(8)) RB2 : virtual RA { 
	__declspec(align(1024)) int b : 3;
};

struct __declspec(align(8)) RB3 : virtual RA { 
	__declspec(align(1024)) int b : 3;
	virtual void f() {}
};

struct RC {
	char _;
	__declspec(align(1024)) int c : 3;
};
struct RE {
	char _;
	RC c;
};
#pragma pack()

// CHECK-LABEL:   0 | struct RB0{{$}}
// CHECK-NEXT:0:0-2 |   int b
// CHECK-NEXT:      | [sizeof=8, align=1024
// CHECK-NEXT:      |  nvsize=4, nvalign=1024]
// CHECK-LABEL:   0 | struct RB1{{$}}
// CHECK-NEXT:    0 |   (RB1 vftable pointer)
// CHECK-NEXT: 1024:0-2 |   int b
// CHECK-NEXT:      | [sizeof=1032, align=1024
// CHECK-NEXT:      |  nvsize=1028, nvalign=1024]
// CHECK-LABEL:   0 | struct RB2{{$}}
// CHECK-NEXT:    0 |   (RB2 vbtable pointer)
// CHECK-NEXT: 1024:0-2 |   int b
// CHECK-NEXT: 1028 |   struct RA (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=1032, align=1024
// CHECK-NEXT:      |  nvsize=1028, nvalign=1024]
// CHECK-LABEL:   0 | struct RB3{{$}}
// CHECK-NEXT:    0 |   (RB3 vftable pointer)
// CHECK-NEXT: 1024 |   (RB3 vbtable pointer)
// CHECK-NEXT: 2048:0-2 |   int b
// CHECK-NEXT: 2052 |   struct RA (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=2056, align=1024
// CHECK-NEXT:      |  nvsize=2052, nvalign=1024]
// CHECK-LABEL:   0 | struct RC{{$}}
// CHECK-NEXT:    0 |   char _
// CHECK-NEXT: 1024:0-2 |   int c
// CHECK-NEXT:      | [sizeof=1028, align=1024
// CHECK-NEXT:      |  nvsize=1028, nvalign=1024]
// CHECK-LABEL:   0 | struct RE{{$}}
// CHECK-NEXT:    0 |   char _
// CHECK-NEXT:    1 |   struct RC c
// CHECK-NEXT:    1 |     char _
// CHECK-NEXT: 1025:0-2 |     int c
// CHECK-NEXT:      | [sizeof=1029, align=1
// CHECK-NEXT:      |  nvsize=1029, nvalign=1]
// CHECK-X64-LABEL:   0 | struct RB0{{$}}
// CHECK-X64-NEXT:    0:0-2 |   int b
// CHECK-X64-NEXT:      | [sizeof=8, align=1024
// CHECK-X64-NEXT:      |  nvsize=4, nvalign=1024]
// CHECK-X64-LABEL:   0 | struct RB1{{$}}
// CHECK-X64-NEXT:    0 |   (RB1 vftable pointer)
// CHECK-X64-NEXT: 1024:0-2 |   int b
// CHECK-X64-NEXT:      | [sizeof=1032, align=1024
// CHECK-X64-NEXT:      |  nvsize=1028, nvalign=1024]
// CHECK-X64-LABEL:   0 | struct RB2{{$}}
// CHECK-X64-NEXT:    0 |   (RB2 vbtable pointer)
// CHECK-X64-NEXT: 1024:0-2 |   int b
// CHECK-X64-NEXT: 1028 |   struct RA (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=1032, align=1024
// CHECK-X64-NEXT:      |  nvsize=1028, nvalign=1024]
// CHECK-X64-LABEL:   0 | struct RB3{{$}}
// CHECK-X64-NEXT:    0 |   (RB3 vftable pointer)
// CHECK-X64-NEXT: 1024 |   (RB3 vbtable pointer)
// CHECK-X64-NEXT: 2048:0-2 |   int b
// CHECK-X64-NEXT: 2052 |   struct RA (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=2056, align=1024
// CHECK-X64-NEXT:      |  nvsize=2052, nvalign=1024]
// CHECK-X64-LABEL:   0 | struct RC{{$}}
// CHECK-X64-NEXT:    0 |   char _
// CHECK-X64-NEXT: 1024:0-2 |   int c
// CHECK-X64-NEXT:      | [sizeof=1028, align=1024
// CHECK-X64-NEXT:      |  nvsize=1028, nvalign=1024]
// CHECK-X64-LABEL:   0 | struct RE{{$}}
// CHECK-X64-NEXT:    0 |   char _
// CHECK-X64-NEXT:    1 |   struct RC c
// CHECK-X64-NEXT:    1 |     char _
// CHECK-X64-NEXT: 1025:0-2 |     int c
// CHECK-X64-NEXT:      | [sizeof=1029, align=1
// CHECK-X64-NEXT:      |  nvsize=1029, nvalign=1]

struct NA {};
struct NB {};
#pragma pack(push, 1)
struct NC : virtual NA, virtual NB {};
#pragma pack(pop)
struct ND : NC {};

// CHECK-LABEL:   0 | struct NA (empty){{$}}
// CHECK-NEXT:      | [sizeof=1, align=1
// CHECK-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-LABEL:   0 | struct NB (empty){{$}}
// CHECK-NEXT:      | [sizeof=1, align=1
// CHECK-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-LABEL:   0 | struct NC{{$}}
// CHECK-NEXT:    0 |   (NC vbtable pointer)
// CHECK-NEXT:    4 |   struct NA (virtual base) (empty)
// CHECK-NEXT:    8 |   struct NB (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=8, align=1
// CHECK-NEXT:      |  nvsize=4, nvalign=1]
// CHECK-LABEL:   0 | struct ND{{$}}
// CHECK-NEXT:    0 |   struct NC (base)
// CHECK-NEXT:    0 |     (NC vbtable pointer)
// CHECK-NEXT:    4 |   struct NA (virtual base) (empty)
// CHECK-NEXT:    8 |   struct NB (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-X64-LABEL:   0 | struct NA (empty){{$}}
// CHECK-X64-NEXT:      | [sizeof=1, align=1
// CHECK-X64-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-X64-LABEL:   0 | struct NB (empty){{$}}
// CHECK-X64-NEXT:      | [sizeof=1, align=1
// CHECK-X64-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-X64-LABEL:   0 | struct NC{{$}}
// CHECK-X64-NEXT:    0 |   (NC vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct NA (virtual base) (empty)
// CHECK-X64-NEXT:   12 |   struct NB (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=12, align=1
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=1]
// CHECK-X64-LABEL:   0 | struct ND{{$}}
// CHECK-X64-NEXT:    0 |   struct NC (base)
// CHECK-X64-NEXT:    0 |     (NC vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct NA (virtual base) (empty)
// CHECK-X64-NEXT:   12 |   struct NB (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=12, align=4
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=4]

struct OA {};
struct OB {};
struct OC : virtual OA, virtual OB {};
#pragma pack(push, 1)
struct OD : OC {};
#pragma pack(pop)

// CHECK-LABEL:   0 | struct OA (empty){{$}}
// CHECK-NEXT:      | [sizeof=1, align=1
// CHECK-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-LABEL:   0 | struct OB (empty){{$}}
// CHECK-NEXT:      | [sizeof=1, align=1
// CHECK-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-LABEL:   0 | struct OC{{$}}
// CHECK-NEXT:    0 |   (OC vbtable pointer)
// CHECK-NEXT:    4 |   struct OA (virtual base) (empty)
// CHECK-NEXT:    8 |   struct OB (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=4, nvalign=4]
// CHECK-LABEL:   0 | struct OD{{$}}
// CHECK-NEXT:    0 |   struct OC (base)
// CHECK-NEXT:    0 |     (OC vbtable pointer)
// CHECK-NEXT:    4 |   struct OA (virtual base) (empty)
// CHECK-NEXT:    8 |   struct OB (virtual base) (empty)
// CHECK-NEXT:      | [sizeof=8, align=1
// CHECK-NEXT:      |  nvsize=4, nvalign=1]
// CHECK-X64-LABEL:   0 | struct OA (empty){{$}}
// CHECK-X64-NEXT:      | [sizeof=1, align=1
// CHECK-X64-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-X64-LABEL:   0 | struct OB (empty){{$}}
// CHECK-X64-NEXT:      | [sizeof=1, align=1
// CHECK-X64-NEXT:      |  nvsize=0, nvalign=1]
// CHECK-X64-LABEL:   0 | struct OC{{$}}
// CHECK-X64-NEXT:    0 |   (OC vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct OA (virtual base) (empty)
// CHECK-X64-NEXT:   12 |   struct OB (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=8]
// CHECK-X64-LABEL:   0 | struct OD{{$}}
// CHECK-X64-NEXT:    0 |   struct OC (base)
// CHECK-X64-NEXT:    0 |     (OC vbtable pointer)
// CHECK-X64-NEXT:    8 |   struct OA (virtual base) (empty)
// CHECK-X64-NEXT:   12 |   struct OB (virtual base) (empty)
// CHECK-X64-NEXT:      | [sizeof=12, align=1
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=1]

struct __declspec(align(4)) PA {
  int c;
};

typedef __declspec(align(8)) PA PB;

#pragma pack(push, 1)
struct PC {
  char a;
  PB x;
};
#pragma pack(pop)

// CHECK:         0 | struct PC
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    8 |   struct PA x
// CHECK-NEXT:    8 |     int c
// CHECK-NEXT:      | [sizeof=16, align=8
// CHECK-NEXT:      |  nvsize=12, nvalign=8]
// CHECK-X64:         0 | struct PC
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:    8 |   struct PA x
// CHECK-X64-NEXT:    8 |     int c
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=12, nvalign=8]

typedef PB PD;

#pragma pack(push, 1)
struct PE {
  char a;
  PD x;
};
#pragma pack(pop)

// CHECK:         0 | struct PE
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    8 |   struct PA x
// CHECK-NEXT:    8 |     int c
// CHECK-NEXT:      | [sizeof=16, align=8
// CHECK-NEXT:      |  nvsize=12, nvalign=8]
// CHECK-X64:         0 | struct PE
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:    8 |   struct PA x
// CHECK-X64-NEXT:    8 |     int c
// CHECK-X64-NEXT:      | [sizeof=16, align=8
// CHECK-X64-NEXT:      |  nvsize=12, nvalign=8]

typedef int __declspec(align(2)) QA;
#pragma pack(push, 1)
struct QB {
  char a;
  QA b;
};
#pragma pack(pop)

// CHECK-LABEL:   0 | struct QB{{$}}
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    2 |   QA b
// CHECK-NEXT:      | [sizeof=6, align=2
// CHECK-NEXT:      |  nvsize=6, nvalign=2]
// CHECK-X64-LABEL:   0 | struct QB{{$}}
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:    2 |   QA b
// CHECK-X64-NEXT:      | [sizeof=6, align=2
// CHECK-X64-NEXT:      |  nvsize=6, nvalign=2]

struct QC {
  char a;
  QA b;
};

// CHECK-LABEL:   0 | struct QC{{$}}
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    4 |   QA b
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct QC{{$}}
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:    4 |   QA b
// CHECK-X64-NEXT:      | [sizeof=8, align=4
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=4]

struct QD {
  char a;
  QA b : 3;
};

// CHECK-LABEL:   0 | struct QD{{$}}
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:4:0-2 |   QA b
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]
// CHECK-X64-LABEL:   0 | struct QD{{$}}
// CHECK-X64-NEXT:    0 |   char a
// CHECK-X64-NEXT:4:0-2 |   QA b
// CHECK-X64-NEXT:      | [sizeof=8, align=4
// CHECK-X64-NEXT:      |  nvsize=8, nvalign=4]

struct __declspec(align(4)) EmptyAlignedLongLongMemb {
  long long FlexArrayMemb[0];
};

// CHECK-LABEL:   0 | struct EmptyAlignedLongLongMemb{{$}}
// CHECK-NEXT:    0 |   long long[0] FlexArrayMemb
// CHECK-NEXT:      | [sizeof=8, align=8
// CHECK-NEXT:      |  nvsize=0, nvalign=8]
// CHECK-X64-LABEL:   0 | struct EmptyAlignedLongLongMemb{{$}}
// CHECK-X64-NEXT:    0 |   long long[0] FlexArrayMemb
// CHECK-X64-NEXT:      | [sizeof=8, align=8
// CHECK-X64-NEXT:      |  nvsize=0, nvalign=8]

#pragma pack(1)
struct __declspec(align(4)) EmptyPackedAlignedLongLongMemb {
  long long FlexArrayMemb[0];
};
#pragma pack()

// CHECK-LABEL:   0 | struct EmptyPackedAlignedLongLongMemb{{$}}
// CHECK-NEXT:    0 |   long long[0] FlexArrayMemb
// CHECK-NEXT:      | [sizeof=4, align=4
// CHECK-NEXT:      |  nvsize=0, nvalign=4]
// CHECK-X64-LABEL:   0 | struct EmptyPackedAlignedLongLongMemb{{$}}
// CHECK-X64-NEXT:    0 |   long long[0] FlexArrayMemb
// CHECK-X64-NEXT:      | [sizeof=4, align=4
// CHECK-X64-NEXT:      |  nvsize=0, nvalign=4]

int a[
sizeof(X)+
sizeof(Y)+
sizeof(Z)+
sizeof(C1)+
sizeof(CA2)+
sizeof(YA)+
sizeof(YB)+
sizeof(YC)+
sizeof(YD)+
sizeof(YE)+
sizeof(YF)+
sizeof(YF)+
sizeof(D2)+
sizeof(JC)+
sizeof(KB)+
sizeof(L)+
sizeof(MB)+
sizeof(RB0)+
sizeof(RB1)+
sizeof(RB2)+
sizeof(RB3)+
sizeof(RC)+
sizeof(RE)+
sizeof(ND)+
sizeof(OD)+
sizeof(PC)+
sizeof(PE)+
sizeof(QB)+
sizeof(QC)+
sizeof(QD)+
sizeof(EmptyAlignedLongLongMemb)+
sizeof(EmptyPackedAlignedLongLongMemb)+
0];
