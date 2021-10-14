// RUN: %clang_cc1 -std=c++2a -fsyntax-only -triple x86_64-linux-gnu -fdump-record-layouts %s | FileCheck %s

namespace Empty {
  struct A {};
  struct B { [[no_unique_address]] A a; char b; };
  static_assert(sizeof(B) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::B
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   char b
  // CHECK-NEXT:       | [sizeof=1, dsize=1, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct C {};
  struct D {
    [[no_unique_address]] A a;
    [[no_unique_address]] C c;
    char d;
  };
  static_assert(sizeof(D) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::D
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   struct Empty::C c (empty)
  // CHECK-NEXT:     0 |   char d
  // CHECK-NEXT:       | [sizeof=1, dsize=1, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct E {
    [[no_unique_address]] A a1;
    [[no_unique_address]] A a2;
    char e;
  };
  static_assert(sizeof(E) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::E
  // CHECK-NEXT:     0 |   struct Empty::A a1 (empty)
  // CHECK-NEXT:     1 |   struct Empty::A a2 (empty)
  // CHECK-NEXT:     0 |   char e
  // CHECK-NEXT:       | [sizeof=2, dsize=2, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct F {
    ~F();
    [[no_unique_address]] A a1;
    [[no_unique_address]] A a2;
    char f;
  };
  static_assert(sizeof(F) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::F
  // CHECK-NEXT:     0 |   struct Empty::A a1 (empty)
  // CHECK-NEXT:     1 |   struct Empty::A a2 (empty)
  // CHECK-NEXT:     0 |   char f
  // CHECK-NEXT:       | [sizeof=2, dsize=1, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct G { [[no_unique_address]] A a; ~G(); };
  static_assert(sizeof(G) == 1);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::G
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:       | [sizeof=1, dsize=0, align=1,
  // CHECK-NEXT:       |  nvsize=1, nvalign=1]

  struct H { [[no_unique_address]] A a, b; ~H(); };
  static_assert(sizeof(H) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::H
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     1 |   struct Empty::A b (empty)
  // CHECK-NEXT:       | [sizeof=2, dsize=0, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct OversizedEmpty : A {
    ~OversizedEmpty();
    [[no_unique_address]] A a;
  };
  static_assert(sizeof(OversizedEmpty) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::OversizedEmpty
  // CHECK-NEXT:     0 |   struct Empty::A (base) (empty)
  // CHECK-NEXT:     1 |   struct Empty::A a (empty)
  // CHECK-NEXT:       | [sizeof=2, dsize=0, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct HasOversizedEmpty {
    [[no_unique_address]] OversizedEmpty m;
  };
  static_assert(sizeof(HasOversizedEmpty) == 2);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::HasOversizedEmpty
  // CHECK-NEXT:     0 |   struct Empty::OversizedEmpty m (empty)
  // CHECK-NEXT:     0 |     struct Empty::A (base) (empty)
  // CHECK-NEXT:     1 |     struct Empty::A a (empty)
  // CHECK-NEXT:       | [sizeof=2, dsize=0, align=1,
  // CHECK-NEXT:       |  nvsize=2, nvalign=1]

  struct EmptyWithNonzeroDSize {
    [[no_unique_address]] A a;
    int x;
    [[no_unique_address]] A b;
    int y;
    [[no_unique_address]] A c;
  };
  static_assert(sizeof(EmptyWithNonzeroDSize) == 12);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::EmptyWithNonzeroDSize
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   int x
  // CHECK-NEXT:     4 |   struct Empty::A b (empty)
  // CHECK-NEXT:     4 |   int y
  // CHECK-NEXT:     8 |   struct Empty::A c (empty)
  // CHECK-NEXT:       | [sizeof=12, dsize=12, align=4,
  // CHECK-NEXT:       |  nvsize=12, nvalign=4]

  struct EmptyWithNonzeroDSizeNonPOD {
    ~EmptyWithNonzeroDSizeNonPOD();
    [[no_unique_address]] A a;
    int x;
    [[no_unique_address]] A b;
    int y;
    [[no_unique_address]] A c;
  };
  static_assert(sizeof(EmptyWithNonzeroDSizeNonPOD) == 12);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct Empty::EmptyWithNonzeroDSizeNonPOD
  // CHECK-NEXT:     0 |   struct Empty::A a (empty)
  // CHECK-NEXT:     0 |   int x
  // CHECK-NEXT:     4 |   struct Empty::A b (empty)
  // CHECK-NEXT:     4 |   int y
  // CHECK-NEXT:     8 |   struct Empty::A c (empty)
  // CHECK-NEXT:       | [sizeof=12, dsize=8, align=4,
  // CHECK-NEXT:       |  nvsize=9, nvalign=4]
}

namespace POD {
  // Cannot reuse tail padding of a PDO type.
  struct A { int n; char c[3]; };
  struct B { [[no_unique_address]] A a; char d; };
  static_assert(sizeof(B) == 12);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct POD::B
  // CHECK-NEXT:     0 |   struct POD::A a
  // CHECK-NEXT:     0 |     int n
  // CHECK-NEXT:     4 |     char [3] c
  // CHECK-NEXT:     8 |   char d
  // CHECK-NEXT:       | [sizeof=12, dsize=12, align=4,
  // CHECK-NEXT:       |  nvsize=12, nvalign=4]
}

namespace NonPOD {
  struct A { int n; char c[3]; ~A(); };
  struct B { [[no_unique_address]] A a; char d; };
  static_assert(sizeof(B) == 8);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct NonPOD::B
  // CHECK-NEXT:     0 |   struct NonPOD::A a
  // CHECK-NEXT:     0 |     int n
  // CHECK-NEXT:     4 |     char [3] c
  // CHECK-NEXT:     7 |   char d
  // CHECK-NEXT:       | [sizeof=8, dsize=8, align=4,
  // CHECK-NEXT:       |  nvsize=8, nvalign=4]
}

namespace NVSizeGreaterThanDSize {
  // The nvsize of an object includes the complete size of its empty subobjects
  // (although it's unclear why). Ensure this corner case is handled properly.
  struct alignas(8) A { ~A(); }; // dsize 0, nvsize 0, size 8
  struct B : A { char c; }; // dsize 1, nvsize 8, size 8
  static_assert(sizeof(B) == 8);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct NVSizeGreaterThanDSize::B
  // CHECK-NEXT:     0 |   struct NVSizeGreaterThanDSize::A (base) (empty)
  // CHECK-NEXT:     0 |   char c
  // CHECK-NEXT:       | [sizeof=8, dsize=1, align=8,
  // CHECK-NEXT:       |  nvsize=8, nvalign=8]

  struct V { int n; };

  // V is at offset 16, not offset 12, because B's tail padding is strangely not
  // usable for virtual bases.
  struct C : B, virtual V {};
  static_assert(sizeof(C) == 24);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct NVSizeGreaterThanDSize::C
  // CHECK-NEXT:     0 |   (C vtable pointer)
  // CHECK-NEXT:     8 |   struct NVSizeGreaterThanDSize::B (base)
  // CHECK-NEXT:     8 |     struct NVSizeGreaterThanDSize::A (base) (empty)
  // CHECK-NEXT:     8 |     char c
  // CHECK-NEXT:    16 |   struct NVSizeGreaterThanDSize::V (virtual base)
  // CHECK-NEXT:    16 |     int n
  // CHECK-NEXT:       | [sizeof=24, dsize=20, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct D : virtual V {
    [[no_unique_address]] B b;
  };
  static_assert(sizeof(D) == 24);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct NVSizeGreaterThanDSize::D
  // CHECK-NEXT:     0 |   (D vtable pointer)
  // CHECK-NEXT:     8 |   struct NVSizeGreaterThanDSize::B b
  // CHECK-NEXT:     8 |     struct NVSizeGreaterThanDSize::A (base) (empty)
  // CHECK-NEXT:     8 |     char c
  // CHECK-NEXT:    16 |   struct NVSizeGreaterThanDSize::V (virtual base)
  // CHECK-NEXT:    16 |     int n
  // CHECK-NEXT:       | [sizeof=24, dsize=20, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]

  struct X : virtual A { [[no_unique_address]] A a; };
  struct E : virtual A {
    [[no_unique_address]] A a;
    // Here, we arrange for X to hang over the end of the nvsize of E. This
    // should force the A vbase to be laid out at offset 24, not 16.
    [[no_unique_address]] X x;
  };
  static_assert(sizeof(E) == 32);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct NVSizeGreaterThanDSize::E
  // CHECK-NEXT:     0 |   (E vtable pointer)
  // CHECK-NEXT:     0 |   struct NVSizeGreaterThanDSize::A a (empty)
  // CHECK-NEXT:     8 |   struct NVSizeGreaterThanDSize::X x
  // CHECK-NEXT:     8 |     (X vtable pointer)
  // CHECK-NEXT:     8 |     struct NVSizeGreaterThanDSize::A a (empty)
  // CHECK-NEXT:    16 |     struct NVSizeGreaterThanDSize::A (virtual base) (empty)
  // CHECK-NEXT:    24 |   struct NVSizeGreaterThanDSize::A (virtual base) (empty)
  // CHECK-NEXT:       | [sizeof=32, dsize=16, align=8,
  // CHECK-NEXT:       |  nvsize=16, nvalign=8]
}

namespace RepeatedVBase {
  struct alignas(16) A { ~A(); };
  struct B : A {};
  struct X : virtual A, virtual B {};
  struct Y { [[no_unique_address]] X x; char c; };
  static_assert(sizeof(Y) == 32);

  // CHECK:*** Dumping AST Record Layout
  // CHECK:          0 | struct RepeatedVBase::Y
  // CHECK-NEXT:     0 |   struct RepeatedVBase::X x
  // CHECK-NEXT:     0 |     (X vtable pointer)
  // CHECK-NEXT:     0 |     struct RepeatedVBase::A (virtual base) (empty)
  // CHECK-NEXT:    16 |     struct RepeatedVBase::B (virtual base) (empty)
  // CHECK-NEXT:    16 |       struct RepeatedVBase::A (base) (empty)
  // CHECK-NEXT:     8 |     char c
  // CHECK-NEXT:       | [sizeof=32, dsize=9, align=16,
  // CHECK-NEXT:       |  nvsize=9, nvalign=16]
}
