// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '+='

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '-='

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '*='

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '/='

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '&='

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '^='

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '<<='

// CHECK: VarDecl
// CHECK-NEXT: Integer
// CHECK-NEXT: CompoundAssignOperator
// CHECK-SAME: '>>='

void expr() {
  f();
}
