// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump %s | FileCheck %s

// CHECK-LABEL: arr_assign1
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr' 'int[2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr2' 'int[2]'
void arr_assign1() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  Arr = Arr2;
}

// CHECK-LABEL: arr_assign2
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr' 'int[2]'
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr2' 'int[2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr3' 'int[2]'
void arr_assign2() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  int Arr3[2] = {2, 2};
  Arr = Arr2 = Arr3;
}

// CHECK-LABEL: arr_assign3
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr' 'int[2][2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr2' 'int[2][2]'
void arr_assign3() {
  int Arr[2][2] = {{0, 1}, {2, 3}};
  int Arr2[2][2] = {{0, 0}, {1, 1}};
  Arr = Arr2;
}

// CHECK-LABEL: arr_assign4
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int' lvalue '='
// CHECK: ArraySubscriptExpr 0x{{[0-9a-f]+}} {{.*}} 'int' lvalue
// CHECK: ImplicitCastExpr 0x{{[0-9a-f]+}} {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK: ParenExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr' 'int[2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr2' 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 6
void arr_assign4() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  (Arr = Arr2)[0] = 6;
}

// CHECK-LABEL: arr_assign5
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int' lvalue '='
// CHECK: ArraySubscriptExpr 0x{{[0-9a-f]+}} {{.*}} 'int' lvalue
// CHECK: ImplicitCastExpr 0x{{[0-9a-f]+}} {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK: ParenExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr' 'int[2]'
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr2' 'int[2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr3' 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 6
void arr_assign5() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  int Arr3[2] = {3, 4};
  (Arr = Arr2 = Arr3)[0] = 6;
}

// CHECK-LABEL: arr_assign6
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int' lvalue '='
// CHECK: ArraySubscriptExpr 0x{{[0-9a-f]+}} {{.*}} 'int' lvalue
// CHECK: ImplicitCastExpr 0x{{[0-9a-f]+}} {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK: ArraySubscriptExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue
// CHECK: ImplicitCastExpr 0x{{[0-9a-f]+}} {{.*}} 'int (*)[2]' <ArrayToPointerDecay>
// CHECK: ParenExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr' 'int[2][2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr2' 'int[2][2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 6
void arr_assign6() {
  int Arr[2][2] = {{0, 1}, {2, 3}};
  int Arr2[2][2] = {{0, 0}, {1, 1}};
  (Arr = Arr2)[0][0] = 6;
}

// CHECK-LABEL: arr_assign7
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: ArraySubscriptExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue
// CHECK: ImplicitCastExpr 0x{{[0-9a-f]+}} {{.*}} 'int (*)[2]' <ArrayToPointerDecay>
// CHECK: ParenExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr' 'int[2][2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var 0x{{[0-9a-f]+}} 'Arr2' 'int[2][2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 6
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 6
void arr_assign7() {
  int Arr[2][2] = {{0, 1}, {2, 3}};
  int Arr2[2][2] = {{0, 0}, {1, 1}};
  (Arr = Arr2)[0] = {6, 6};
}
