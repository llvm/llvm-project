// Each function in a TU tests a specific graph topology. The driver
// function connects them via return values.
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

// RUN: %clang_cc1 -fsyntax-only %t/driver.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/driver.json

// ============================================================================
// Step 2: Link TU summaries.
// ============================================================================

// RUN: clang-ssaf-linker \
// RUN:   %t/simple.json %t/chain.json %t/fan_out.json \
// RUN:   %t/fan_in.json %t/disconnected.json %t/cycle.json \
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

// RUN: %clang_cc1 -fsyntax-only %t/driver.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=DRIVER

//--- simple.cpp
// Unsafe use on q.
int *simple_unsafe(int *q) {
  q[5];
  return q;
}
// SIMPLE: unsafe pointer level:(q, 1)
// SIMPLE: #unsafe pointer level: 1

//--- chain.cpp
int *chain(int *p) {
  // ret -> l -> p
  int *l, *ret;
  l = p;
  ret = l;
  return ret;
}
// CHAIN-DAG: unsafe pointer level:(ret, 1)
// CHAIN-DAG: unsafe pointer level:(l, 1)
// CHAIN-DAG: unsafe pointer level:(p, 1)
// CHAIN: #unsafe pointer level: 3

//--- fan_out.cpp
int *fan_out(int *center) {
  int *l1, *l2, *l3;

  center = l1;
  center = l2;
  center = l3;
  return l1;
}
// FAN-OUT: unsafe pointer level:(l1, 1)
// FAN-OUT: #unsafe pointer level: 1

//--- fan_in.cpp
int *fan_in(int *center) {
  int *l1, *l2, *l3;

  l1 = center;
  l2 = center;
  l3 = center;
  return l1;
}
// FAN-IN-DAG: unsafe pointer level:(center, 1)
// FAN-IN-DAG: unsafe pointer level:(l1, 1)
// FAN-IN: #unsafe pointer level: 2

//--- disconnected.cpp
int *disconnected(int *p, int *q) {
  return p;
}
// DISCONNECTED: unsafe pointer level:(p, 1)
// DISCONNECTED: #unsafe pointer level: 1

//--- cycle.cpp
int *cycle(int *p) {
  int *l1, *l2, *l3;
  p = l3;
  l1 = p;
  l2 = l1;
  l3 = l2;
  return l2;
}
// CYCLE-DAG: unsafe pointer level:(p, 1)
// CYCLE-DAG: unsafe pointer level:(l1, 1)
// CYCLE-DAG: unsafe pointer level:(l2, 1)
// CYCLE-DAG: unsafe pointer level:(l3, 1)
// CYCLE: #unsafe pointer level: 4

//--- driver.cpp
int *simple_unsafe(int *);
int *chain(int *);
int *fan_out(int *);
int *fan_in(int *);
int *disconnected(int *, int *);
int *cycle(int *);

void driver(int *p) {
  simple_unsafe(p);  // p is unsafe

  int *a, *b, *c, *d, *e, *f;
                          // abbreviated "return value" as "p" below:
  p = chain(a);           // p -> ... -> a, so a is unsafe
  p = fan_out(b);         // p is a fan-out leaf and b is the center, b NOT propagated
  p = fan_in(c);          // p is a fan-in leaf and c is the center, so c is unsafe
  p = disconnected(d, e); // p -> d; e is disconnected, so only d is unsafe 
  p = cycle(f);           // p and f are in a cycle, so f is unsafe
}
// DRIVER-DAG: unsafe pointer level:(p, 1)
// DRIVER-DAG: unsafe pointer level:(a, 1)
// DRIVER-DAG: unsafe pointer level:(c, 1)
// DRIVER-DAG: unsafe pointer level:(d, 1)
// DRIVER-DAG: unsafe pointer level:(f, 1)
// DRIVER: #unsafe pointer level: 5
