// RUN: rm -rf %t && mkdir -p %t


// RUN: %clang_cc1 -fsyntax-only %s \
// RUN:   --ssaf-extract-summaries=PointerFlow,UnsafeBufferUsage \
// RUN:   --ssaf-tu-summary-file=%t/tu.summary.json \
// RUN:   --ssaf-compilation-unit-id="tu-1"


// RUN: clang-ssaf-linker %t/tu.summary.json -o %t/lu.json


// RUN: clang-ssaf-analyzer %t/lu.json -o %t/wpa.json \
// RUN:   -a UnsafeBufferReachableAnalysisResult

// RUN: FileCheck %s --input-file=%t/wpa.json

extern int **G;

void foo(int *p = G[5]); // (p, 1) -> (G, 2) and G is unsafe

void foo(int *p) {
  int *q = p;    // (q, 1) -> (p, 1)
  q[5] = 0;      // q is unsafe
}

// Check that (q, 1), (p, 1), (G, 2) and (G, 1) are all unsafe pointers.

// CHECK-DAG: "id": [[P_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "1",[[:space:]]+"usr": }}"c:@F@foo#*I#"
// CHECK-DAG: "id": [[G_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "",[[:space:]]+"usr": }}"c:@G"
// CHECK-DAG: "id": [[Q_ID:[0-9]+]],{{([^]]|[[:space:]])+\],[[:space:]]+"suffix": "",[[:space:]]+"usr": "[^"]+@q"}}

// CHECK: "analysis_name": "UnsafeBufferReachableAnalysisResult"

// CHECK-DAG: {{\{[[:space:]]+}}"@": [[G_ID]]{{[[:space:]]+\},[[:space:]]+1[[:space:]]+\]}}
// CHECK-DAG: {{\{[[:space:]]+}}"@": [[G_ID]]{{[[:space:]]+\},[[:space:]]+2[[:space:]]+\]}}
// CHECK-DAG: {{\{[[:space:]]+}}"@": [[Q_ID]]{{[[:space:]]+\},[[:space:]]+1[[:space:]]+\]}}
// CHECK-DAG: {{\{[[:space:]]+}}"@": [[P_ID]]{{[[:space:]]+\},[[:space:]]+1[[:space:]]+\]}}

// CHECK: "analysis_name":
