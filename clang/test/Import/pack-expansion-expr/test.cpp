// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: PackExpansionExpr
// CHECK-SAME: '<dependent type>'
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: 'T...'
// CHECK-SAME: ParmVar
// CHECK-SAME: 'a'

void expr() {
  f();
}
