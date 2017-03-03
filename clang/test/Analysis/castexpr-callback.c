// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyzer-config debug.AnalysisOrder:PreStmtCastExpr=true,debug.AnalysisOrder:PostStmtCastExpr=true %s 2>&1 | FileCheck %s

void test(char c) {
  int i = (int)c;
}

// CHECK: PreStmt<CastExpr> (Kind : LValueToRValue)
// CHECK-NEXT: PostStmt<CastExpr> (Kind : LValueToRValue)
// CHECK-NEXT: PreStmt<CastExpr> (Kind : IntegralCast)
// CHECK-NEXT: PostStmt<CastExpr> (Kind : IntegralCast)
