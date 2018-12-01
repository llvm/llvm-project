// RUN: %clang_cc1 -Wno-unused -fblocks -fobjc-exceptions -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

void TestBlockExpr(int x) {
  ^{ x; };
}
// CHECK:      FunctionDecl{{.*}}TestBlockExpr
// CHECK:      BlockExpr{{.*}} 'void (^)(void)'
// CHECK-NEXT:   BlockDecl

void TestExprWithCleanup(int x) {
  ^{ x; };
}
// CHECK:      FunctionDecl{{.*}}TestExprWithCleanup
// CHECK:      ExprWithCleanups
// CHECK-NEXT:   cleanup Block
// CHECK-NEXT:   BlockExpr

@interface A
@end

void TestObjCAtCatchStmt() {
  @try {
  } @catch(A *a) {
  } @catch(...) {
  } @finally {
  }
}
// CHECK:      FunctionDecl{{.*}}TestObjCAtCatchStmt
// CHECK:      ObjCAtTryStmt
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:   ObjCAtCatchStmt{{.*}}
// CHECK-NEXT:     VarDecl{{.*}}a
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   ObjCAtCatchStmt{{.*}} catch all
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   ObjCAtFinallyStmt
