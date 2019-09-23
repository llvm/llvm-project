// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

template<int N = 0>
void pr43370() {
  int arr[2];
  __atomic_store_n(arr, 0, 5);
}

template<int N = 0>
void foo() {
  int arr[2];
  (void)__atomic_compare_exchange_n(arr, arr, 1, 0, 3, 4);
}

void useage(){
  pr43370();
  foo();
}

// CHECK:FunctionTemplateDecl 0x{{[0-9a-f]+}} <{{[^,]+}}, line:7:1> line:4:6 pr43370
// CHECK: AtomicExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <{{[^:]+}}:20> 'int [2]' lvalue Var 0x{{[0-9a-f]+}} 'arr' 'int [2]'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:28> 'int' 5
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:25> 'int' 0
// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <line:4:1, line:7:1> line:4:6 used pr43370
// CHECK: AtomicExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <{{[^:]+}}:20> 'int [2]' lvalue Var 0x{{[0-9a-f]+}} 'arr' 'int [2]'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:28> 'int' 5
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:25> 'int' 0

// CHECK:FunctionTemplateDecl 0x{{[0-9a-f]+}} <line:9:1, line:13:1> line:10:6 foo
// CHECK: AtomicExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <{{[^:]+}}:37> 'int [2]' lvalue Var 0x{{[0-9a-f]+}} 'arr' 'int [2]'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:53> 'int' 3
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <{{[^:]+}}:42> 'int [2]' lvalue Var 0x{{[0-9a-f]+}} 'arr' 'int [2]'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:56> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:47> 'int' 1
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:50> 'int' 0
// CHECK:FunctionDecl 0x{{[0-9a-f]+}} <line:10:1, line:13:1> line:10:6 used foo
// CHECK: AtomicExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <{{[^:]+}}:37> 'int [2]' lvalue Var 0x{{[0-9a-f]+}} 'arr' 'int [2]'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:53> 'int' 3
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: <ArrayToPointerDecay>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-f]+}} <{{[^:]+}}:42> 'int [2]' lvalue Var 0x{{[0-9a-f]+}} 'arr' 'int [2]'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:56> 'int' 4
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:47> 'int' 1
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-f]+}} <{{[^:]+}}:50> 'int' 0
