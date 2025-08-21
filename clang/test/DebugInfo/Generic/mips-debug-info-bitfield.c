// RUN: %clang_cc1 -x c -debug-info-kind=limited -triple mips-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct fields
{
  unsigned a : 4;
  unsigned b : 4;
} flags;

// CHECK: !DIDerivedType(tag: DW_TAG_member,
// CHECK-SAME: {{.*}}name: "a"
// CHECK-NOT: {{.*}}offset:
// CHECK-SAME: {{.*}}flags: DIFlagBitField

// CHECK: !DIDerivedType(tag: DW_TAG_member,
// CHECK-SAME: {{.*}}name: "b"
// CHECK-SAME: {{.*}}offset: 4
// CHECK-SAME: {{.*}}flags: DIFlagBitField
