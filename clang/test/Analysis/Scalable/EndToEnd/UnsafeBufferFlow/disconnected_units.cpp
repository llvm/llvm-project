// Each function in a TU tests a specific graph topology.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// ============================================================================
// Step 1: Extract TU summaries from each TU.
// ============================================================================

// RUN: %clang_cc1 -fsyntax-only %t/simple.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/simple.json

// RUN: %clang_cc1 -fsyntax-only %t/chain.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/chain.json

// RUN: %clang_cc1 -fsyntax-only %t/fan_out.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/fan_out.json

// RUN: %clang_cc1 -fsyntax-only %t/fan_in.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/fan_in.json

// RUN: %clang_cc1 -fsyntax-only %t/disconnected.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/disconnected.json

// RUN: %clang_cc1 -fsyntax-only %t/cycle.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/cycle.json

// ============================================================================
// Step 2: Link TU summaries.
// ============================================================================

// RUN: clang-ssaf-linker \
// RUN:   %t/simple.json %t/chain.json %t/fan_out.json \
// RUN:   %t/fan_in.json %t/disconnected.json %t/cycle.json \
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

// RUN: %clang_cc1 -fsyntax-only %t/simple.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=SIMPLE

// RUN: %clang_cc1 -fsyntax-only %t/chain.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHAIN

// RUN: %clang_cc1 -fsyntax-only %t/fan_out.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=FAN-OUT

// RUN: %clang_cc1 -fsyntax-only %t/fan_in.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=FAN-IN

// RUN: %clang_cc1 -fsyntax-only %t/disconnected.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=DISCONNECTED

// RUN: %clang_cc1 -fsyntax-only %t/cycle.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=CYCLE

//--- simple.cpp
void simple(int *q) {
  q[5];
}
// SIMPLE: unsafe pointer level:(q, 1)
// SIMPLE: #unsafe pointer level: 1

//--- chain.cpp
void chain(int *a) {
  int *b = a;
  int *c = b;
  c[5];
}
// CHAIN-DAG: unsafe pointer level:(a, 1)
// CHAIN-DAG: unsafe pointer level:(b, 1)
// CHAIN-DAG: unsafe pointer level:(c, 1)
// CHAIN: #unsafe pointer level: 3

//--- fan_out.cpp
void fan_out(int *s1, int *s2) {
  int *dst;
  dst = s1;
  dst = s2;
  dst[2];
}
// FAN-OUT-DAG: unsafe pointer level:(s1, 1)
// FAN-OUT-DAG: unsafe pointer level:(s2, 1)
// FAN-OUT-DAG: unsafe pointer level:(dst, 1)
// FAN-OUT: #unsafe pointer level: 3

//--- fan_in.cpp
void fan_in(int *src) {
  int *d1 = src;
  int *d2 = src;
  int *d3 = src;
  src[1];
}
// FAN-IN: unsafe pointer level:(src, 1)
// FAN-IN: #unsafe pointer level: 1

//--- disconnected.cpp
void disconnected(int *safe, int *unsafe_ptr) {
  *safe = 42;
  unsafe_ptr[3];
}
// DISCONNECTED: unsafe pointer level:(unsafe_ptr, 1)
// DISCONNECTED: #unsafe pointer level: 1

//--- cycle.cpp
void cycle(int *x) {
  int *y = x;
  int *z = y;
  x = z;
  z[4];
}
// CYCLE-DAG: unsafe pointer level:(x, 1)
// CYCLE-DAG: unsafe pointer level:(y, 1)
// CYCLE-DAG: unsafe pointer level:(z, 1)
// CYCLE: #unsafe pointer level: 3
