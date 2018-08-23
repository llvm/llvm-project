// RUN: clang-import-test -x objective-c++ -Xcc -fobjc-exceptions -dump-ast -import %S/Inputs/F.m -expression %s | FileCheck %s

// FIXME: Seems that Objective-C try/catch crash codegen on Windows. Reenable once this is fixed.
// UNSUPPORTED: system-windows

// CHECK: ObjCAtTryStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: ObjCAtThrowStmt
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-NEXT: ObjCAtCatchStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname
// CHECK-SAME: 'Exception *'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ObjCAtFinallyStmt
// CHECK-NEXT: CompoundStmt

// CHECK-NEXT: ObjCAtTryStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ObjCAtCatchStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname1
// CHECK-SAME: 'Exception *'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ObjCAtThrowStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: ObjCAtCatchStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname2
// CHECK-SAME: 'OtherException *'
// CHECK-NEXT: CompoundStmt

// CHECK-NEXT: ObjCAtTryStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ObjCAtFinallyStmt
// CHECK-NEXT: CompoundStmt

void expr() {
  f();
}
