// RUN:  %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon  -target-feature +fp8 \
// RUN:  -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s
#include<arm_neon.h>

void test_locals(void) {
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "__MFloat8_t", {{.*}}, baseType: ![[ELTTYU8:[0-9]+]]
  // CHECK-DAG: ![[ELTTYU8]] = !DIBasicType(name: "__MFloat8_t", size: 8, encoding: DW_ATE_unsigned_char)
  __MFloat8_t mfp8;
}
