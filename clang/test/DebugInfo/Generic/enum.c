// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "e"
// CHECK-SAME:             elements: [[TEST3_ENUMS:![0-9]*]]
// CHECK: [[TEST3_ENUMS]] = !{[[TEST3_E:![0-9]*]]}
// CHECK: [[TEST3_E]] = !DIEnumerator(name: "E", value: -1)

enum e;
void func(enum e *p) {
}
enum e { E = -1 };
