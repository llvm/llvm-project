// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK-C
// RUN: %clang_cc1 -emit-llvm -o - %s -x c++ | FileCheck %s --check-prefixes=CHECK-CXX

// CHECK-LABEL: @initlist_atomic_pointer_zero
struct range_tree_node_s {
  long base;
  long left;
  struct range_tree_node_s *_Atomic right;
};

void initlist_atomic_pointer_zero() {
  (struct range_tree_node_s){.right = 0};
}

// CHECK-LABEL: @scalar_atomic_init
void scalar_atomic_init() {
  // CHECK-C: store ptr null
  // CHECK-CXX: store ptr null
  _Atomic(int*) p = 0;
}
