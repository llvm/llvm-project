// RUN:  %clang_cc1 -triple aarch64-none-linux-gnu -target-feature -fp8 \
// RUN:  -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

// REQUIRES: aarch64-registered-target

void test_locals(void) {
  // CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "__mfp8", {{.*}}, baseType: ![[ELTTYU8:[0-9]+]]
  // CHECK-DAG: ![[ELTTYU8]] = !DIBasicType(name: "__mfp8", size: 8, encoding: DW_ATE_unsigned_char)
  __mfp8 mfp8;
}
