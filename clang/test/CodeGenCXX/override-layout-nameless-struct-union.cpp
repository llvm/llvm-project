// RUN: %clang_cc1 -w -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-layout-nameless-struct-union.layout %s | FileCheck %s

// CHECK: Type: struct S
// CHECK:   Size:64
// CHECK:   Alignment:32
// CHECK:   FieldOffsets: [0, 32, 32]
struct S {
  short _s;
//union {
    int _su0;
    char _su1;
//};
};

// CHECK: Type: union U
// CHECK:   Size:96
// CHECK:   Alignment:32
// CHECK:   FieldOffsets: [0, 0, 32, 64, 68, 73]
union U {
  short _u;
//struct {
    char _us0;
    int _us1;
    unsigned _us20 : 4;
    unsigned _us21 : 5;
    unsigned _us22 : 6;
//};
};

void use_structs() {
  S ss[sizeof(S)];
  U us[sizeof(U)];
}
