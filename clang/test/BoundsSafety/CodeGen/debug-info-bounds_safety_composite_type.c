
// RUN: %clang_cc1 -fbounds-safety -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "__bounds_safety::wide_ptr

#include <ptrcheck.h>

void f() {
  int arr[10];
  int *ptr = &arr[0];
}

int test_attrs(int *__counted_by(num_elements - 1) ptr_counted_by,
               int *__sized_by(num_elements * 4) ptr_sized_by, int *end,
               int *__ended_by(end) ptr_ended_by, int num_elements) {
  return 0;
}

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__bounds_safety::counted_by::num_elements - 1"
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__bounds_safety::sized_by::num_elements * 4"
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__bounds_safety::dynamic_range::ptr_ended_by::"
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "__bounds_safety::dynamic_range::::end"
