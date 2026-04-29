// TUs are connected via function call arguments forming a tree.
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

// RUN: %clang_cc1 -fsyntax-only %t/link.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/link.json

// RUN: %clang_cc1 -fsyntax-only %t/branch1.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/branch1.json

// RUN: %clang_cc1 -fsyntax-only %t/branch2.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/branch2.json

// RUN: %clang_cc1 -fsyntax-only %t/driver.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/driver.json


// ============================================================================
// Step 2: Link TU summaries.
// ============================================================================

// RUN: clang-ssaf-linker \
// RUN:   %t/unsafe.json %t/link.json %t/branch1.json \
// RUN:   %t/branch2.json %t/driver.json \
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

// RUN: %clang_cc1 -fsyntax-only %t/link.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=LINK

// RUN: %clang_cc1 -fsyntax-only %t/branch1.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=BRANCH1

// RUN: %clang_cc1 -fsyntax-only %t/branch2.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=BRANCH2

// RUN: %clang_cc1 -fsyntax-only %t/driver.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=DRIVER

//--- unsafe.cpp
void unsafe_use(int *p) {
  p[5];
}
// UNSAFE: unsafe pointer level:(p, 1)
// UNSAFE: #unsafe pointer level: 1

//--- link.cpp
void unsafe_use(int *);
void link(int *a) {
  unsafe_use(a);
}
// LINK: unsafe pointer level:(a, 1)
// LINK: #unsafe pointer level: 1

//--- branch1.cpp
void link(int *);
void branch1(int *b) {
  link(b);
}
// BRANCH1: unsafe pointer level:(b, 1)
// BRANCH1: #unsafe pointer level: 1

//--- branch2.cpp
void link(int *);
void branch2(int *c) {
  int *l;
  l = c;
  link(l);
}
// BRANCH2-DAG: unsafe pointer level:(c, 1)
// BRANCH2-DAG: unsafe pointer level:(l, 1)
// BRANCH2: #unsafe pointer level: 2

//--- driver.cpp
void branch1(int *);
void branch2(int *);
void driver() {
  int *x, *y;
  branch1(x);
  branch2(y);
}
// DRIVER-DAG: unsafe pointer level:(x, 1)
// DRIVER-DAG: unsafe pointer level:(y, 1)
// DRIVER: #unsafe pointer level: 2
