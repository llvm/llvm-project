// RUN: %clang_cc1 -std=c++2a -fsyntax-only -triple x86_64-windows-msvc -fms-compatibility -fdump-record-layouts %s | FileCheck %s

namespace Empty {
  struct A {};
  struct A2 {};
  struct A3 { [[msvc::no_unique_address]] A a; };
  struct alignas(8) A4 {};

  struct B {
    [[msvc::no_unique_address]] A a;
    char b;
  };
  static_assert(sizeof(B) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::B
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   char b
  // CHECK-NEXT:       | [sizeof=1, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct C {
    [[msvc::no_unique_address]] A a;
    [[msvc::no_unique_address]] A2 a2;
    char c;
  };
  static_assert(sizeof(C) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::C
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   struct Empty::A2 a2 (empty)
  // CHECK-NEXT:     0 |   char c
  // CHECK-NEXT:       | [sizeof=1, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct D {
    [[msvc::no_unique_address]] A3 a;
    int i;
  };
  static_assert(sizeof(D) == 8);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::D
  // CHECK-NEXT:     0 |   struct Empty::A3 a (empty)
  // CHECK-NEXT:     0 |     struct Empty::A a (empty)
  // CHECK-NEXT:     4 |   int i
  // CHECK-NEXT:       | [sizeof=8, align=4,
  // CHECK-NEXT:       |  nvsize=8, nvalign=4]

  struct E {
    [[msvc::no_unique_address]] A a1;
    [[msvc::no_unique_address]] A a2;
    char e;
  };
  static_assert(sizeof(E) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::E
  // CHECK-NEXT:     0 |   struct Empty::A a1 (empty)
  // CHECK-NEXT:     1 |   struct Empty::A a2 (empty)
  // CHECK-NEXT:     0 |   char e
  // CHECK-NEXT:       | [sizeof=2, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct F {
    ~F();
    [[msvc::no_unique_address]] A a1;
    [[msvc::no_unique_address]] A a2;
    char f;
  };
  static_assert(sizeof(F) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::F
  // CHECK-NEXT:     0 |   struct Empty::A a1 (empty)
  // CHECK-NEXT:     1 |   struct Empty::A a2 (empty)
  // CHECK-NEXT:     0 |   char f
  // CHECK-NEXT:       | [sizeof=2, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct G { [[msvc::no_unique_address]] A a; ~G(); };
  static_assert(sizeof(G) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::G
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:       | [sizeof=1, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct H {
    [[msvc::no_unique_address]] A a;
    [[msvc::no_unique_address]] A b;
    ~H();
  };
  static_assert(sizeof(H) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::H
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     1 |   struct Empty::A b (empty)
  // CHECK-NEXT:       | [sizeof=2, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct I {
    [[msvc::no_unique_address]] A4 a;
    [[msvc::no_unique_address]] A4 b;
  };
  static_assert(sizeof(I) == 16);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::I
  // CHECK-NEXT:     0 |   struct Empty::A4 a (empty)
  // CHECK-NEXT:     8 |   struct Empty::A4 b (empty)
  // CHECK-NEXT:       | [sizeof=16, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct J {
    [[msvc::no_unique_address]] A4 a;
    A4 b;
  };
  static_assert(sizeof(J) == 16);

  // MSVC puts a and b at the same offset.
  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::J
  // CHECK-NEXT:     0 |   struct Empty::A4 a (empty)
  // CHECK-NEXT:     8 |   struct Empty::A4 b (empty)
  // CHECK-NEXT:       | [sizeof=16, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct K {
    [[msvc::no_unique_address]] A4 a;
    [[msvc::no_unique_address]] char c;
    [[msvc::no_unique_address]] A4 b;
  };
  static_assert(sizeof(K) == 16);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::K
  // CHECK-NEXT:     0 |   struct Empty::A4 a (empty)
  // CHECK-NEXT:     0 |   char c
  // CHECK-NEXT:     8 |   struct Empty::A4 b (empty)
  // CHECK-NEXT:       | [sizeof=16, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct OversizedEmpty : A {
    ~OversizedEmpty();
    [[msvc::no_unique_address]] A a;
  };
  static_assert(sizeof(OversizedEmpty) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::OversizedEmpty
  // CHECK-NEXT:     0 |   struct Empty::A (base) (empty)
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:       | [sizeof=1, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct HasOversizedEmpty {
    [[msvc::no_unique_address]] OversizedEmpty m;
  };
  static_assert(sizeof(HasOversizedEmpty) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::HasOversizedEmpty
  // CHECK-NEXT:     0 |   struct Empty::OversizedEmpty m (empty)
  // CHECK-NEXT:     0 |     struct Empty::A (base) (empty)
  // CHECK-NEXT:     0 |     struct Empty::A a (empty)
  // CHECK-NEXT:       | [sizeof=1, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct EmptyWithNonzeroDSize {
    [[msvc::no_unique_address]] A a;
    int x;
    [[msvc::no_unique_address]] A b;
    int y;
    [[msvc::no_unique_address]] A c;
  };
  static_assert(sizeof(EmptyWithNonzeroDSize) == 8);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::EmptyWithNonzeroDSize
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   int x
  // CHECK-NEXT:     1 |   struct Empty::A b (empty)
  // CHECK-NEXT:     4 |   int y
  // CHECK-NEXT:     2 |   struct Empty::A c (empty)
  // CHECK-NEXT:       | [sizeof=8,  align=4,
  // CHECK-NEXT:       |  nvsize=8, nvalign=4]

  struct EmptyWithNonzeroDSizeNonPOD {
    ~EmptyWithNonzeroDSizeNonPOD();
    [[msvc::no_unique_address]] A a;
    int x;
    [[msvc::no_unique_address]] A b;
    int y;
    [[msvc::no_unique_address]] A c;
  };
  static_assert(sizeof(EmptyWithNonzeroDSizeNonPOD) == 8);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::EmptyWithNonzeroDSizeNonPOD
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   int x
  // CHECK-NEXT:     1 |   struct Empty::A b (empty)
  // CHECK-NEXT:     4 |   int y
  // CHECK-NEXT:     2 |   struct Empty::A c (empty)
  // CHECK-NEXT:       | [sizeof=8, align=4,
  // CHECK-NEXT:       |  nvsize=8, nvalign=4]
}

namespace POD {
  struct A { int n; char c[3]; };
  struct B { [[msvc::no_unique_address]] A a; char d; };
  static_assert(sizeof(B) == 12);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct POD::B
  // CHECK-NEXT:     0 |   struct POD::A a
  // CHECK-NEXT:     0 |     int n
  // CHECK-NEXT:     4 |     char[3] c
  // CHECK-NEXT:     8 |   char d
  // CHECK-NEXT:       | [sizeof=12,  align=4,
  // CHECK-NEXT:       |  nvsize=12, nvalign=4]
}

namespace NonPOD {
  struct A { int n; char c[3]; ~A(); };
  struct B { [[msvc::no_unique_address]] A a; char d; };
  static_assert(sizeof(B) == 12);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct NonPOD::B
  // CHECK-NEXT:     0 |   struct NonPOD::A a
  // CHECK-NEXT:     0 |     int n
  // CHECK-NEXT:     4 |     char[3] c
  // CHECK-NEXT:     8 |   char d
  // CHECK-NEXT:       | [sizeof=12, align=4,
  // CHECK-NEXT:       |  nvsize=12, nvalign=4]
}

namespace VBases {
  // The nvsize of an object includes the complete size of its empty subobjects
  // (although it's unclear why). Ensure this corner case is handled properly.
  struct Empty {};
  struct alignas(8) A {}; // dsize 0, nvsize 0, size 8
  struct B : A { char c; }; // dsize 1, nvsize 8, size 8
  static_assert(sizeof(B) == 8);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct VBases::B
  // CHECK-NEXT:     0 |   struct VBases::A (base) (empty)
  // CHECK-NEXT:     0 |   char c
  // CHECK-NEXT:       | [sizeof=8, align=8,
  // CHECK-NEXT:       |  nvsize=8, nvalign=8]

  struct V { int n; };

  struct C : B, virtual V {};
  static_assert(sizeof(C) == 24);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct VBases::C
  // CHECK-NEXT:     0 |   struct VBases::B (base)
  // CHECK-NEXT:     0 |     struct VBases::A (base) (empty)
  // CHECK-NEXT:     0 |     char c
  // CHECK-NEXT:     8 |   (C vbtable pointer)
  // CHECK-NEXT:    16 |   struct VBases::V (virtual base)
  // CHECK-NEXT:    16 |     int n
  // CHECK-NEXT:       | [sizeof=24, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct D : virtual Empty {
    [[msvc::no_unique_address]] Empty a;
  };
  static_assert(sizeof(D) == 16);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct VBases::D
  // CHECK-NEXT:     0 |   (D vbtable pointer)
  // CHECK-NEXT:     8 |   struct VBases::Empty a
  // CHECK-NEXT:    16 |   struct VBases::Empty (virtual base) (empty)
  // CHECK-NEXT:       | [sizeof=16, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct E : virtual V {
    [[msvc::no_unique_address]] B b;
  };
  static_assert(sizeof(E) == 24);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct VBases::E
  // CHECK-NEXT:     0 |   (E vbtable pointer)
  // CHECK-NEXT:     8 |   struct VBases::B b
  // CHECK-NEXT:     8 |     struct VBases::A (base) (empty)
  // CHECK-NEXT:     8 |     char c
  // CHECK-NEXT:    16 |   struct VBases::V (virtual base)
  // CHECK-NEXT:    16 |     int n
  // CHECK-NEXT:       | [sizeof=24, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct X : virtual A { [[msvc::no_unique_address]] A a; };
  struct F : virtual A {
    [[msvc::no_unique_address]] A a;
    [[msvc::no_unique_address]] X x;
  };
  static_assert(sizeof(F) == 24);

  // MSVC places x after a and the total size is 48.
  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct VBases::F
  // CHECK-NEXT:     0 |   (F vbtable pointer)
  // CHECK-NEXT:     8 |   struct VBases::A a (empty)
  // CHECK-NEXT:     8 |   struct VBases::X x
  // CHECK-NEXT:     8 |     (X vbtable pointer)
  // CHECK-NEXT:    16 |     struct VBases::A a (empty)
  // CHECK-NEXT:    24 |     struct VBases::A (virtual base) (empty)
  // CHECK-NEXT:    24 |   struct VBases::A (virtual base) (empty)
  // CHECK-NEXT:       | [sizeof=24, align=8,
  // CHECK-NEXT:       |  nvsize=24, nvalign=8]

  struct G : virtual Empty {
    int i;
    [[msvc::no_unique_address]] A a;
  };
  static_assert(sizeof(G) == 16);

  // MSVC places a at offset 12.
  // CHECK:*** Dumping AST Record Layout
  // CHECK:         0 | struct VBases::G
  // CHECK-NEXT:    0 |   (G vbtable pointer)
  // CHECK-NEXT:    8 |   int i
  // CHECK-NEXT:    8 |   struct VBases::A a (empty)
  // CHECK-NEXT:   16 | struct VBases::Empty (virtual base) (empty)
  // CHECK-NEXT:      | [sizeof=16, align=8,
  // CHECK-NEXT:      |  nvsize=16, nvalign=8]
}

namespace ZeroSize {
  struct empty {};

  union empty_union {};

  struct empty_union_container {
    [[msvc::no_unique_address]] empty_union x;
  };

  union union_of_empty {
    [[msvc::no_unique_address]] empty x;
  };

  struct struct_of_empty {
    [[msvc::no_unique_address]] empty x;
  };

  struct union_of_empty_container {
    [[msvc::no_unique_address]] union_of_empty x;
  };
  static_assert(sizeof(union_of_empty_container) == 1);
  // CHECK:*** Dumping AST Record Layout
  // CHECK:         0 | struct ZeroSize::union_of_empty_container
  // CHECK-NOT: (empty)
  // CHECK:         0 |   union ZeroSize::union_of_empty x (empty)
  // CHECK:         0 |     struct ZeroSize::empty x (empty)
  // CHECK:           | [sizeof=1, align=1,
  // CHECK:           |  nvsize=1, nvalign=1] 

  struct struct_of_empty_container {
    [[msvc::no_unique_address]] struct_of_empty x;
  }; 
  static_assert(sizeof(struct_of_empty_container) == 1);
  // CHECK:*** Dumping AST Record Layout
  // CHECK:         0 | struct ZeroSize::struct_of_empty_container
  // CHECK-NOT: (empty)
  // CHECK:         0 |   struct ZeroSize::struct_of_empty x (empty)
  // CHECK:         0 |     struct ZeroSize::empty x (empty)
  // CHECK:           | [sizeof=1, align=1,
  // CHECK:           |  nvsize=1, nvalign=1] 

}
