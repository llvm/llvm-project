// RUN: %clang_cc1 -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts-simple -foverride-record-layout=%S/Inputs/override-layout-virtual-base.layout %s | FileCheck %s

struct S1 {
  int a;
};

struct S2 : virtual S1 {
  virtual void foo() {}
};

// CHECK: Type: struct S3
// CHECK:   FieldOffsets: [128]
struct S3 : S2 {
  char b;
};

void use_structs() {
  S1 s1s[sizeof(S1)];
  S2 s2s[sizeof(S2)];
  S3 s3s[sizeof(S3)];
}
