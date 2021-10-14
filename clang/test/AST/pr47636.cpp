// RUN: %clang_cc1 -fsyntax-only %s -ast-dump | FileCheck %s

int(&&intu_rvref)[] {1,2,3,4};
// CHECK: VarDecl 0x[[GLOB_ADDR:[0-9a-f]+]] {{.*}} intu_rvref 'int (&&)[4]' listinit
// CHECK-NEXT: ExprWithCleanups {{.*}} 'int [4]' xvalue
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'int [4]' xvalue extended by Var 0x[[GLOB_ADDR]] 'intu_rvref' 'int (&&)[4]'
// CHECK-NEXT: InitListExpr {{.*}} 'int [4]'

// CHECK: FunctionDecl {{.*}} static_const
void static_const() {
  static const int(&&intu_rvref)[] {1,2,3,4};
  // CHECK: VarDecl 0x[[STATIC_ADDR:[0-9a-f]+]] {{.*}} intu_rvref 'const int (&&)[4]' static listinit
  // CHECK-NEXT: ExprWithCleanups {{.*}} 'const int [4]' xvalue
  // CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'const int [4]' xvalue extended by Var 0x[[STATIC_ADDR]] 'intu_rvref' 'const int (&&)[4]'
  // CHECK-NEXT: InitListExpr {{.*}} 'const int [4]'
}

// CHECK: FunctionDecl {{.*}} const_expr
constexpr int const_expr() {
  int(&&intu_rvref)[]{1, 2, 3, 4};
  // CHECK: VarDecl 0x[[CE_ADDR:[0-9a-f]+]] {{.*}} intu_rvref 'int (&&)[4]' listinit
  // CHECK-NEXT: ExprWithCleanups {{.*}} 'int [4]' xvalue
  // CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'int [4]' xvalue extended by Var 0x[[CE_ADDR]] 'intu_rvref' 'int (&&)[4]'
  // CHECK-NEXT: InitListExpr {{.*}} 'int [4]'
  return intu_rvref[0];
}
