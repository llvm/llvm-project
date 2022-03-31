// Test without serialization:
// RUN: %clang_cc1 -Wno-unused -fobjc-arc -fblocks -fobjc-exceptions -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -Wno-unused -fobjc-arc -fblocks -fobjc-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -x objective-c -Wno-unused -fobjc-arc -fblocks -fobjc-exceptions -include-pch %t \
// RUN: -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck -strict-whitespace %s

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

void TestObjCAtCatchStmt(void) {
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

typedef struct {
  id f;
} S;

id TestCompoundLiteral(id a) {
  return ((S){ .f = a }).f;
}

// CHECK:     FunctionDecl{{.*}}TestCompoundLiteral
// CHECK:       ExprWithCleanups
// CHECK-NEXT:    cleanup CompoundLiteralExpr
// CHECK:           CompoundLiteralExpr{{.*}}'S':'S' lvalue
