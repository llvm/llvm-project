// RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyzer-config debug.AnalysisOrder:PreStmtOffsetOfExpr=true,debug.AnalysisOrder:PostStmtOffsetOfExpr=true %s 2>&1 | FileCheck %s
#include "Inputs/system-header-simulator.h"

struct S {
  char c;
};

void test() {
  offsetof(struct S, c); 
}

// CHECK: PreStmt<OffsetOfExpr>
// CHECK-NEXT: PostStmt<OffsetOfExpr>