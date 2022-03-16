// RUN: %clang_cc1 -triple x86_64-linux-pc -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
typedef _Bool bool512 __attribute__((ext_vector_type(512)));

bool512 b;

// Test that we get bit-sized bool elements on x86
// CHECK: !DICompositeType(tag: DW_TAG_array_type,
// CHECK-SAME:             baseType: ![[BOOL:[0-9]+]]
// CHECK-SAME:             size: 512
// CHECK-SAME:             DIFlagVector
// CHECK: ![[BOOL]] = !DIBasicType(name: "char"
