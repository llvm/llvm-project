// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: CXXDynamicCastExpr
// CHECK-SAME: dynamic_cast
// CHECK-SAME: <Dynamic>

// CHECK: CXXStaticCastExpr
// CHECK-SAME: static_cast
// CHECK-SAME: <BaseToDerived (A)>

// CHECK: CXXReinterpretCastExpr
// CHECK-SAME: reinterpret_cast
// CHECK-SAME: <BitCast>

// CHECK: CXXConstCastExpr
// CHECK-SAME: const_cast
// CHECK-SAME: <NoOp>

void expr() {
  f();
}
