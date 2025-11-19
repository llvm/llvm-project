// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

enum { A = 1 };
int func1(int a) {
  switch(a) {
  case A: return 10;
  default: break;
  }
  return 0;
}
// CHECK:       !DICompositeType(tag: DW_TAG_enumeration_type
// CHECK-SAME:  elements: [[TEST1_ENUMS:![0-9]*]]
// CHECK:       [[TEST1_ENUMS]] = !{[[TEST1_E:![0-9]*]]}
// CHECK:       [[TEST1_E]] = !DIEnumerator(name: "A", value: 1)

// Test ImplicitCast of switch case enum value
enum { B = 2 };
typedef unsigned long long __t1;
typedef __t1 __t2;
int func2(__t2 a) {
  switch(a) {
  case B: return 10;
  default: break;
  }
  return 0;
}
// CHECK:       !DICompositeType(tag: DW_TAG_enumeration_type
// CHECK-SAME:  elements: [[TEST2_ENUMS:![0-9]*]]
// CHECK:       [[TEST2_ENUMS]] = !{[[TEST2_E:![0-9]*]]}
// CHECK:       [[TEST2_E]] = !DIEnumerator(name: "B", value: 2)
