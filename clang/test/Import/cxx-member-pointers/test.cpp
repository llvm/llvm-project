// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s

// CHECK: VarDecl
// CHECK-SAME: int S::*
// CHECK-NEXT: CallExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-SAME: int S::*(*)()
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: int S::*()

void expr() {
  int S::*p = iptr();
  S s;
  s.i = 3;
  int i = s.*p;
}
