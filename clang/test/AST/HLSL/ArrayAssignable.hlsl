// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump %s | FileCheck %s

// CHECK-LABEL: arr_assign1
// CHECK: CompoundStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: DeclStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: VarDecl [[A:0x[0-9a-f]+]] {{.*}} col:7 used Arr 'int[2]' cinit
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 1
// CHECK: DeclStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: VarDecl [[B:0x[0-9a-f]+]] {{.*}} col:7 used Arr2 'int[2]' cinit
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var [[A]] 'Arr' 'int[2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var [[B]] 'Arr2' 'int[2]'
void arr_assign1() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  Arr = Arr2;
}

// CHECK-LABEL: arr_assign2
// CHECK: CompoundStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: DeclStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: VarDecl [[A:0x[0-9a-f]+]] {{.*}} col:7 used Arr 'int[2]' cinit
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 1
// CHECK: DeclStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: VarDecl [[B:0x[0-9a-f]+]] {{.*}} col:7 used Arr2 'int[2]' cinit
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: DeclStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: VarDecl [[C:0x[0-9a-f]+]] {{.*}} col:7 used Arr3 'int[2]' cinit
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 2
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 2
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var [[A]] 'Arr' 'int[2]'
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var [[B]] 'Arr2' 'int[2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]' lvalue Var [[C]] 'Arr3' 'int[2]'
void arr_assign2() {
  int Arr[2] = {0, 1};
  int Arr2[2] = {0, 0};
  int Arr3[2] = {2, 2};
  Arr = Arr2 = Arr3;
}

// CHECK-LABEL: arr_assign3
// CHECK: CompoundStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: DeclStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: VarDecl [[A:0x[0-9a-f]+]] {{.*}} col:7 used Arr 'int[2][2]' cinit
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]'
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 1
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 2
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 3
// CHECK: DeclStmt 0x{{[0-9a-f]+}} {{.*}}
// CHECK: VarDecl [[B:0x[0-9a-f]+]] {{.*}} col:7 used Arr2 'int[2][2]' cinit
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]'
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 0
// CHECK: InitListExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2]'
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 1
// CHECK: IntegerLiteral 0x{{[0-9a-f]+}} {{.*}} 'int' 1
// CHECK: BinaryOperator 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue '='
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var [[A]] 'Arr' 'int[2][2]'
// CHECK: DeclRefExpr 0x{{[0-9a-f]+}} {{.*}} 'int[2][2]' lvalue Var [[B]] 'Arr2' 'int[2][2]'
void arr_assign3() {
  int Arr[2][2] = {{0, 1}, {2, 3}};
  int Arr2[2][2] = {{0, 0}, {1, 1}};
  Arr = Arr2;
}