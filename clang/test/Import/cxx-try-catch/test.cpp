// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: CXXTryStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXCatchStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: CompoundStmt

// CHECK: CXXTryStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXCatchStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: 'int'
// CHECK-NEXT: CompoundStmt

// CHECK: CXXTryStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXCatchStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname
// CHECK-SAME: 'int'
// CHECK-NEXT: CompoundStmt

// CHECK: CXXTryStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXCatchStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname1
// CHECK-SAME: 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXCatchStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname2
// CHECK-SAME: 'long'
// CHECK-NEXT: CompoundStmt

void expr() {
  f();
}
