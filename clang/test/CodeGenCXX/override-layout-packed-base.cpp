// RUN: %clang_cc1 -triple i686-windows-msvc -w -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-layout-packed-base.layout %s | FileCheck %s

//#pragma pack(push, 1)

// CHECK: Type: class B<0>
// CHECK:   Size:40
// CHECK:   FieldOffsets: [0, 32]

// CHECK: Type: class B<1>
// CHECK:   Size:40
// CHECK:   FieldOffsets: [0, 32]

template<int I>
class B {
  int _b1;
  char _b2;
};

// CHECK: Type: class C
// CHECK:   Size:88
// CHECK:   FieldOffsets: [80]

class C : B<0>, B<1> {
  char _c;
};

// CHECK: Type: class D
// CHECK:   Size:120
// CHECK:   FieldOffsets: [32]

class D : virtual B<0>, virtual B<1> {
  char _d;
};

//#pragma pack(pop)

void use_structs() {
  C cs;
  D ds;
}
