// RUN: %clang_cc1 -x c -debug-info-kind=limited -emit-llvm -triple x86_64-apple-darwin -o - %s | FileCheck %s

// CHECK: %struct.layout3 = type <{ i8, [3 x i8], %struct.size8_pack4, i8, [3 x i8] }>
// CHECK: %struct.layout0 = type { i8, %struct.size8, i8 }
// CHECK: %struct.layout1 = type <{ i8, %struct.size8_anon, i8, [2 x i8] }>
// CHECK: %struct.layout2 = type <{ i8, %struct.size8_pack1, i8 }>

// ---------------------------------------------------------------------
// Not packed.
// ---------------------------------------------------------------------
struct size8 {
  int i : 4;
  long long l : 60;
};
struct layout0 {
  char l0_ofs0;
  struct size8 l0_ofs8;
  int l0_ofs16 : 1;
};
// CHECK: l0_ofs0
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs8",
// CHECK-SAME:     {{.*}}size: 64, offset: 64)
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs16",
// CHECK-SAME:     {{.*}}size: 1, offset: 128, flags: DIFlagBitField, extraData: i64 128)


// ---------------------------------------------------------------------
// Implicitly packed.
// ---------------------------------------------------------------------
struct size8_anon {
  int : 4;
  long long : 60;
};
struct layout1 {
  char l1_ofs0;
  struct size8_anon l1_ofs1;
  int l1_ofs9 : 1;
};
// CHECK: l1_ofs0
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs1",
// CHECK-SAME:     {{.*}}size: 64, offset: 8)
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs9",
// CHECK-SAME:     {{.*}}size: 1, offset: 72, flags: DIFlagBitField, extraData: i64 72)


// ---------------------------------------------------------------------
// Explicitly packed.
// ---------------------------------------------------------------------
#pragma pack(1)
struct size8_pack1 {
  int i : 4;
  long long l : 60;
};
struct layout2 {
  char l2_ofs0;
  struct size8_pack1 l2_ofs1;
  int l2_ofs9 : 1;
};
#pragma pack()
// CHECK: l2_ofs0
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs1",
// CHECK-SAME:     {{.*}}size: 64, offset: 8)
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs9",
// CHECK-SAME:     {{.*}}size: 1, offset: 72, flags: DIFlagBitField, extraData: i64 72)



// ---------------------------------------------------------------------
// Explicitly packed with different alignment.
// ---------------------------------------------------------------------
#pragma pack(4)
struct size8_pack4 {
  int i : 4;
  long long l : 60;
};
struct layout3 {
  char l3_ofs0;
  struct size8_pack4 l3_ofs4;
  int l3_ofs12 : 1;
};
#pragma pack()
// CHECK: l3_ofs0
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs4",
// CHECK-SAME:     {{.*}}size: 64, offset: 32)
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs12",
// CHECK-SAME:     {{.*}}size: 1, offset: 96, flags: DIFlagBitField, extraData: i64 96)

struct layout3 l3;
struct layout0 l0;
struct layout1 l1;
struct layout2 l2;
