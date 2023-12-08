// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
typedef int v4si __attribute__((__vector_size__(16)));

v4si a;

// Test that we get an array type that's also a vector out of debug.
// CHECK: !DICompositeType(tag: DW_TAG_array_type,
// CHECK-SAME:             baseType: ![[INT:[0-9]+]]
// CHECK-SAME:             size: 128
// CHECK-SAME:             DIFlagVector
// CHECK: ![[INT]] = !DIBasicType(name: "int"
