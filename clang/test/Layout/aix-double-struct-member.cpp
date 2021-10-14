// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack %s | \
// RUN:   FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -fdump-record-layouts \
// RUN:     -fsyntax-only -fxl-pragma-pack %s | \
// RUN:   FileCheck %s

namespace test1 {
// Test the class layout when having a double which is/is not the first struct
// member.
struct D {
  double d1;
  int i1;
};

struct DoubleFirst {
  struct D d2;
  int i2;
};

struct IntFirst {
  int i3;
  struct D d3;
};

int a = sizeof(DoubleFirst);
int b = sizeof(IntFirst);

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:         0 | struct test1::D
// CHECK-NEXT:         0 |   double d1
// CHECK-NEXT:         8 |   int i1
// CHECK-NEXT:           | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:           |  nvsize=16, nvalign=4, preferrednvalign=8]

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:         0 | struct test1::DoubleFirst
// CHECK-NEXT:         0 |   struct test1::D d2
// CHECK-NEXT:         0 |     double d1
// CHECK-NEXT:         8 |     int i1
// CHECK-NEXT:        16 |   int i2
// CHECK-NEXT:           | [sizeof=24, dsize=24, align=4, preferredalign=8,
// CHECK-NEXT:           |  nvsize=24, nvalign=4, preferrednvalign=8]

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:         0 | struct test1::IntFirst
// CHECK-NEXT:         0 |   int i3
// CHECK-NEXT:         4 |   struct test1::D d3
// CHECK-NEXT:         4 |     double d1
// CHECK-NEXT:        12 |     int i1
// CHECK-NEXT:           | [sizeof=20, dsize=20, align=4, preferredalign=4,
// CHECK-NEXT:           |  nvsize=20, nvalign=4, preferrednvalign=4]
} // namespace test1

namespace test2 {
// Test the class layout when having a zero-sized bitfield followed by double.
struct Double {
  int : 0;
  double d;
};

int a = sizeof(Double);

// CHECK:     *** Dumping AST Record Layout
// CHECK-NEXT:         0 | struct test2::Double
// CHECK-NEXT:       0:- |   int
// CHECK-NEXT:         0 |   double d
// CHECK-NEXT:           | [sizeof=8, dsize=8, align=4, preferredalign=4,
// CHECK-NEXT:           |  nvsize=8, nvalign=4, preferrednvalign=4]
} // namespace test2

namespace test3 {
// Test the class layout when having a double member in union.
union A {
  int b;
  double d;
};

struct UnionStruct {
  union A a;
  int i;
};

int a = sizeof(UnionStruct);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | union test3::A
// CHECK-NEXT:          0 |   int b
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test3::UnionStruct
// CHECK-NEXT:          0 |   union test3::A a
// CHECK-NEXT:          0 |     int b
// CHECK-NEXT:          0 |     double d
// CHECK-NEXT:          8 |   int i
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

} // namespace test3

namespace test4 {
// Test the class layout when having multiple base classes.
struct A {
  int a;
};

struct B {
  double d;
};

class S : A, B {
};

int a = sizeof(S);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::A
// CHECK-NEXT:          0 |   int a
// CHECK-NEXT:            | [sizeof=4, dsize=4, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4, preferrednvalign=4]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test4::B
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | class test4::S
// CHECK-NEXT:          0 |   struct test4::A (base)
// CHECK-NEXT:          0 |     int a
// CHECK-NEXT:          4 |   struct test4::B (base)
// CHECK-NEXT:          4 |     double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]
} // namespace test4

namespace test5 {
struct Empty {
};

struct EmptyDer : Empty {
  double d;
};

struct NonEmpty {
  int i;
};

struct NonEmptyDer : NonEmpty {
  double d;
};

int a = sizeof(EmptyDer);
int b = sizeof(NonEmptyDer);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::Empty (empty)
// CHECK-NEXT:            | [sizeof=1, dsize=1, align=1, preferredalign=1,
// CHECK-NEXT:            |  nvsize=1, nvalign=1, preferrednvalign=1]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::EmptyDer
// CHECK-NEXT:          0 |   struct test5::Empty (base) (empty)
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::NonEmpty
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:            | [sizeof=4, dsize=4, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=4, nvalign=4, preferrednvalign=4]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test5::NonEmptyDer
// CHECK-NEXT:          0 |   struct test5::NonEmpty (base)
// CHECK-NEXT:          0 |     int i
// CHECK-NEXT:          4 |   double d
// CHECK-NEXT:            | [sizeof=12, dsize=12, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=12, nvalign=4, preferrednvalign=4]
} // namespace test5

namespace test6 {
struct A {
  struct B {
    double d[3];
  } b;
};

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test6::A::B
// CHECK-NEXT:          0 |   double [3] d
// CHECK-NEXT:            | [sizeof=24, dsize=24, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=24, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test6::A
// CHECK-NEXT:          0 |   struct test6::A::B b
// CHECK-NEXT:          0 |     double [3] d
// CHECK-NEXT:            | [sizeof=24, dsize=24, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=24, nvalign=4, preferrednvalign=8]

} // namespace test6

namespace test7 {
struct A {
  struct B {
    long double _Complex d[3];
  } b;
};

int a = sizeof(A);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test7::A::B
// CHECK-NEXT:          0 |   _Complex long double [3] d
// CHECK-NEXT:            | [sizeof=48, dsize=48, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=48, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test7::A
// CHECK-NEXT:          0 |   struct test7::A::B b
// CHECK-NEXT:          0 |     _Complex long double [3] d
// CHECK-NEXT:            | [sizeof=48, dsize=48, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=48, nvalign=4, preferrednvalign=8]

} // namespace test7

namespace test8 {
struct Emp {};

struct Y : Emp {
  double d;
};

struct Z : Emp {
  Y y;
};

int a = sizeof(Z);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test8::Emp (empty)
// CHECK-NEXT:            | [sizeof=1, dsize=1, align=1, preferredalign=1,
// CHECK-NEXT:            |  nvsize=1, nvalign=1, preferrednvalign=1]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test8::Y
// CHECK-NEXT:          0 |   struct test8::Emp (base) (empty)
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test8::Z
// CHECK-NEXT:          0 |   struct test8::Emp (base) (empty)
// CHECK-NEXT:          8 |   struct test8::Y y
// CHECK-NEXT:          8 |     struct test8::Emp (base) (empty)
// CHECK-NEXT:          8 |     double d
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

} // namespace test8

namespace test9 {
// Test the class layout when having a zero-extent array in a base class, which
// renders the base class not empty.
struct A { char zea[0]; };

struct B : A { double d; };

struct C { double d; };
struct D : A, C { char x; };

int a = sizeof(B);
int b = sizeof(D);

// CHECK:               0 | struct test9::B
// CHECK-NEXT:          0 |   struct test9::A (base)
// CHECK-NEXT:          0 |     char [0] zea
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=4]

// CHECK:               0 | struct test9::D
// CHECK-NEXT:          0 |   struct test9::A (base)
// CHECK-NEXT:          0 |     char [0] zea
// CHECK-NEXT:          0 |   struct test9::C (base)
// CHECK-NEXT:          0 |     double d
// CHECK-NEXT:          8 |   char x
// CHECK-NEXT:            | [sizeof=12, dsize=9, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=9, nvalign=4, preferrednvalign=4]

} // namespace test9

namespace test10 {
struct A { double x; };
struct B : A {};

int a = sizeof(B);

// CHECK:               0 | struct test10::B
// CHECK-NEXT:          0 |   struct test10::A (base)
// CHECK-NEXT:          0 |     double x
// CHECK-NEXT:            | [sizeof=8, dsize=8, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=8, nvalign=4, preferrednvalign=8]

} // namespace test10

namespace test11 {
// Test how #pragma pack and align attribute interacts with AIX `power`
// alignment rules.
struct A {
  char a;
  double __attribute__((aligned(16))) d;
  int i;
};

struct B {
  double __attribute__((aligned(4))) d1;
  char a;
  double d2;
};

#pragma pack(2)
struct C {
  int i;
  short j;
  double k;
};
#pragma pack(pop)

#pragma pack(2)
struct D {
  double d;
  short j;
  int i;
};
#pragma pack(pop)

#pragma pack(8)
struct E {
  double __attribute__((aligned(4))) d;
  short s;
};
#pragma pack(pop)

#pragma pack(4)
struct F : public D {
  double d;
};
#pragma pack(pop)

#pragma pack(2)
struct G : public E {
  int i;
};
#pragma pack(pop)

int a = sizeof(A);
int b = sizeof(B);
int c = sizeof(C);
int d = sizeof(D);
int e = sizeof(E);
int f = sizeof(F);
int g = sizeof(G);

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test11::A
// CHECK-NEXT:          0 |   char a
// CHECK-NEXT:         16 |   double d
// CHECK-NEXT:         24 |   int i
// CHECK-NEXT:            | [sizeof=32, dsize=32, align=16, preferredalign=16,
// CHECK-NEXT:            |  nvsize=32, nvalign=16, preferrednvalign=16]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test11::B
// CHECK-NEXT:          0 |   double d1
// CHECK-NEXT:          8 |   char a
// CHECK-NEXT:         12 |   double d2
// CHECK-NEXT:            | [sizeof=24, dsize=24, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=24, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test11::C
// CHECK-NEXT:          0 |   int i
// CHECK-NEXT:          4 |   short j
// CHECK-NEXT:          6 |   double k
// CHECK-NEXT:            | [sizeof=14, dsize=14, align=2, preferredalign=2,
// CHECK-NEXT:            |  nvsize=14, nvalign=2, preferrednvalign=2]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test11::D
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:          8 |   short j
// CHECK-NEXT:         10 |   int i
// CHECK-NEXT:            | [sizeof=14, dsize=14, align=2, preferredalign=2,
// CHECK-NEXT:            |  nvsize=14, nvalign=2, preferrednvalign=2]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test11::E
// CHECK-NEXT:          0 |   double d
// CHECK-NEXT:          8 |   short s
// CHECK-NEXT:            | [sizeof=16, dsize=16, align=4, preferredalign=8,
// CHECK-NEXT:            |  nvsize=16, nvalign=4, preferrednvalign=8]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test11::F
// CHECK-NEXT:          0 |   struct test11::D (base)
// CHECK-NEXT:          0 |     double d
// CHECK-NEXT:          8 |     short j
// CHECK-NEXT:         10 |     int i
// CHECK-NEXT:         16 |   double d
// CHECK-NEXT:            | [sizeof=24, dsize=24, align=4, preferredalign=4,
// CHECK-NEXT:            |  nvsize=24, nvalign=4, preferrednvalign=4]

// CHECK:      *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct test11::G
// CHECK-NEXT:          0 |   struct test11::E (base)
// CHECK-NEXT:          0 |     double d
// CHECK-NEXT:          8 |     short s
// CHECK-NEXT:         16 |   int i
// CHECK-NEXT:            | [sizeof=20, dsize=20, align=2, preferredalign=2,
// CHECK-NEXT:            |  nvsize=20, nvalign=2, preferrednvalign=2]

} // namespace test11
