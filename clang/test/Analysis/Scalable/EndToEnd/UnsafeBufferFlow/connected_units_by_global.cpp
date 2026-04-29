// TUs are connected via global variables forming a simple chain.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// ============================================================================
// Step 1: Extract TU summaries from each TU.
// ============================================================================

// RUN: %clang_cc1 -fsyntax-only %t/unsafe.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/unsafe.json

// RUN: %clang_cc1 -fsyntax-only %t/link1.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/link1.json

// RUN: %clang_cc1 -fsyntax-only %t/link2.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/link2.json

// RUN: %clang_cc1 -fsyntax-only %t/driver.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/driver.json

// ============================================================================
// Step 2: Link TU summaries.
// ============================================================================

// RUN: clang-ssaf-linker \
// RUN:   %t/unsafe.json %t/link1.json %t/link2.json \
// RUN:   %t/driver.json \
// RUN:   -o %t/lu-summary.json

// ============================================================================
// Step 3: Run PointerFlowReachable analysis.
// ============================================================================

// RUN: clang-ssaf-analyzer %t/lu-summary.json \
// RUN:   -o %t/wpa-result.json \
// RUN:   -a UnsafeBufferReachableAnalysisResult

// ============================================================================
// Step 4: Apply WPA results to each TU.
// ============================================================================

// RUN: %clang_cc1 -fsyntax-only %t/unsafe.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=UNSAFE

// RUN: %clang_cc1 -fsyntax-only %t/link1.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=LINK1

// RUN: %clang_cc1 -fsyntax-only %t/link2.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=LINK2

// RUN: %clang_cc1 -fsyntax-only %t/driver.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=DRIVER

//--- unsafe.cpp
extern int *g0;
void unsafe_use() {
  g0[5];
}
// UNSAFE: unsafe pointer level:(g0, 1)
// UNSAFE: #unsafe pointer level: 1

//--- link1.cpp
extern int *g0, *g1;
void link1() {
  g0 = g1;
}
// LINK1: unsafe pointer level:(g1, 1)
// LINK1: #unsafe pointer level: 1

//--- link2.cpp
extern int *g1, *g2, *g3;
void link2() {
  int *l;
  l = g2;
  g2 = g1;
  g1 = g3;
  g3 = l;
}
// LINK2-DAG: unsafe pointer level:(g1, 1)
// LINK2-DAG: unsafe pointer level:(g2, 1)
// LINK2-DAG: unsafe pointer level:(g3, 1)
// LINK2-DAG: unsafe pointer level:(l, 1)
// LINK2: #unsafe pointer level: 4

//--- driver.cpp
int *g0, *g1, *g2, *g3;
void unsafe_use();
void link1();
void link2();

void driver(int *p) {
  p = g0;        // p flows into g0
  unsafe_use();  // g0 unsafe
  link1();       // g0 -> g1
  link2();       // local var -> g2 -> g1 -> g3 -> local var
}
// DRIVER-DAG: unsafe pointer level:(g0, 1)
// DRIVER-DAG: unsafe pointer level:(g1, 1)
// DRIVER-DAG: unsafe pointer level:(g2, 1)
// DRIVER-DAG: unsafe pointer level:(g3, 1)
// DRIVER: #unsafe pointer level: 4
