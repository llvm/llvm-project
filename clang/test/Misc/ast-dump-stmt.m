// RUN: %clang_cc1 -Wno-unused -fblocks -fobjc-exceptions -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

void TestBlockExpr(int x) {
  ^{ x; };
}
// CHECK:      Dumping TestBlockExpr
// CHECK:      BlockExpr{{.*}} decl=
// CHECK-NEXT:   capture ParmVar
// CHECK-NEXT:   CompoundStmt

void TestExprWithCleanup(int x) {
  ^{ x; };
}
// CHECK:      Dumping TestExprWithCleanup
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
// CHECK:      Dumping TestObjCAtCatchStmt
// CHECK:      ObjCAtTryStmt
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:   ObjCAtCatchStmt{{.*}} catch parm = "A *a"
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   ObjCAtCatchStmt{{.*}} catch all
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   ObjCAtFinallyStmt
