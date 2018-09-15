// RUN: %clang_cc1 -w -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-layout-packed-base.layout %s | FileCheck %s

// CHECK: Type: class B<0>
// CHECK:   FieldOffsets: [0, 32]

// CHECK: Type: class B<1>
// CHECK:   FieldOffsets: [0, 32]

//#pragma pack(push, 1)
template<int I>
class B {
  int _b1;
  char _b2;
};
//#pragma pack(pop)

// CHECK: Type: class C
// CHECK:   FieldOffsets: [80]

class C : B<0>, B<1> {
  char _c;
};

void use_structs() {
  C cs[sizeof(C)];
}
