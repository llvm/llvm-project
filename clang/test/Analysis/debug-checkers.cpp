// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpDominators %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=DOM-CHECK
// DOM-CHECK: Immediate dominance tree (Node#,IDom#)

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpPostDominators %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=POSTDOM-CHECK
// POSTDOM-CHECK: Immediate post dominance tree (Node#,IDom#)

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpControlDependencies %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=CTRLDEPS-CHECK
// CTRLDEPS-CHECK: Control dependencies (Node#,Dependency#)

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpLiveVars %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=LIVE-VARS-CHECK
// LIVE-VARS-CHECK: live variables at block exit

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpLiveExprs %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=LIVE-EXPRS-CHECK
// LIVE-EXPRS-CHECK: live expressions at block exit

// Skip testing CFGViewer.

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpCFG %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=CFG-CHECK
// CFG-CHECK: ENTRY

// Skip testing CallGraphViewer.

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.DumpCallGraph %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=CALL-GRAPH-CHECK
// CALL-GRAPH-CHECK: --- Call graph Dump ---

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ConfigDumper %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=CONFIG-CHECK
// CONFIG-CHECK: [config]

// Skip testing ExplodedGraphViewer.

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ReportStmts %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s -check-prefix=REPORT-STMTS-CHECK
// REPORT-STMTS-CHECK: warning: Statement

void foo(int *p) {
  *p = 3;
}

int bar() {
  int x;
  foo(&x);
  return x;
}
