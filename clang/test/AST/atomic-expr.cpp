// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

template<int N = 0>
void pr43370() {
  int arr[2];
  __atomic_store_n(arr, 0, 0);
}
void useage(){
  pr43370();
}

// CHECK:FunctionTemplateDecl 0x{{[0-9a-f]+}} <{{[^,]+}}, line:7:1> line:4:6 pr43370
// CHECK: AtomicExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <line:4:1, line:7:1> line:4:6 used pr43370
// CHECK: AtomicExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
