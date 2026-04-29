// Class definition in header, implementation and client in separate TUs.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// ============================================================================
// Step 1: Extract TU summaries from each TU.
// ============================================================================

// RUN: %clang_cc1 -fsyntax-only -I %t %t/impl.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/impl.json

// RUN: %clang_cc1 -fsyntax-only -I %t %t/client.cpp \
// RUN:   --ssaf-extract-summaries=UnsafeBufferUsage,PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/client.json

// ============================================================================
// Step 2: Link TU summaries.
// ============================================================================

// RUN: clang-ssaf-linker \
// RUN:   %t/impl.json %t/client.json \
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

// RUN: %clang_cc1 -fsyntax-only -I %t %t/impl.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=IMPL

// RUN: %clang_cc1 -fsyntax-only -I %t %t/client.cpp \
// RUN:   --ssaf-apply-source-pass=UnsafeBufferReachableDebugAnalysis --ssaf-load-wpa-result=%t/wpa-result.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=CLIENT

//--- myclass.h
struct MyClass {
  int *buf;
  int *safe_ptr;
  void init(int *p);
  void unsafe_op();
  void safe_op(int *s);
};

//--- impl.cpp
#include "myclass.h"
void MyClass::init(int *p) {
  buf = p;
}
void MyClass::unsafe_op() {
  buf[5];
}
void MyClass::safe_op(int *s) {
  safe_ptr = s;
}
// IMPL-DAG: unsafe pointer level:(MyClass::buf, 1)
// IMPL-DAG: unsafe pointer level:(p, 1)
// IMPL: #unsafe pointer level: 2

//--- client.cpp
#include "myclass.h"
void client(int * p, int *q, int *r) {
  MyClass obj;
  obj.init(q);
  obj.unsafe_op();
  obj.safe_op(r);
  obj.buf = p;
}
// CLIENT-DAG: unsafe pointer level:(q, 1)
// CLIENT-DAG: unsafe pointer level:(p, 1)
// CLIENT-DAG: unsafe pointer level:(MyClass::buf, 1)
// CLIENT: #unsafe pointer level: 3
