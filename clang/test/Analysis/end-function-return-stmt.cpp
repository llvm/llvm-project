//RUN: %clang_analyze_cc1 -analyzer-checker=debug.AnalysisOrder -analyzer-config debug.AnalysisOrder:EndFunction=true %s 2>&1 | FileCheck %s

// At the end of a function, we can only obtain a ReturnStmt if the last
// CFGElement in the CFGBlock is either a CFGStmt or a CFGAutomaticObjDtor.

void noReturnStmt() {}

struct S {
  S();
  ~S();
};

int dtorAfterReturnStmt() {
  S s;
  return 0;
}

S endsWithReturnStmt() {
  return S();
}

// endsWithReturnStmt()
// CHECK:      EndFunction
// CHECK-NEXT: ReturnStmt: yes
// CHECK-NEXT: CFGElement: CFGStmt

// dtorAfterReturnStmt()
// CHECK:      EndFunction
// CHECK-NEXT: ReturnStmt: yes
// CHECK-NEXT: CFGElement: CFGAutomaticObjDtor

// noReturnStmt()
// CHECK:      EndFunction
// CHECK-NEXT: ReturnStmt: no
