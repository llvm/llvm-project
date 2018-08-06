// RUN: %clang_cc1 -w -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-bit-field-layout.layout %s | FileCheck %s

// CHECK: Type: struct S1
// CHECK:   FieldOffsets: [0, 11]
struct S1 {
  short a : 3;
  short b : 5;
};

// CHECK: Type: struct S2
// CHECK:   FieldOffsets: [64]
struct S2 {
  virtual ~S2() = default;
  short a : 3;
};

void use_structs() {
  S1 s1s[sizeof(S1)];
  S2 s2s[sizeof(S2)];
}
