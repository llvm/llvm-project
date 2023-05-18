// RUN: %clang_cc1 %std_cxx98-14 -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-bit-field-layout.layout %s | FileCheck %s --check-prefixes=CHECK,PRE17
// RUN: %clang_cc1 -std=c++17 -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-bit-field-layout.layout %s | FileCheck %s --check-prefixes=CHECK,CXX17
// RUN: %clang_cc1 %std_cxx20- -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-bit-field-layout.layout %s | FileCheck %s --check-prefixes=CHECK,CXX20

// PRE17: Type: struct S1
// PRE17:   FieldOffsets: [0, 11]
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

// CXX17: Type: struct S1
// CXX17:   FieldOffsets: [0, 11]
// CXX20: Type: struct S1
// CXX20:   FieldOffsets: [0, 3]

// CHECK: Type: struct S3
// CHECK:   Size:32
// CHECK:   FieldOffsets: [0, 1]
struct S3 {
  int a : 1;
  int b : 2;
};

// CHECK: Type: struct S4
// CHECK:   FieldOffsets: [32]
struct S4 : S3 {
  char c;
};

void use_structs() {
  S1 s1s[sizeof(S1)];
  S2 s2s[sizeof(S2)];
  S3 s3s[sizeof(S3)];
  S4 s4s[sizeof(S4)];
}
