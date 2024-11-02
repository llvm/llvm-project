// RUN: %clang_analyze_cc1 -analyzer-checker=core %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK

// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPRECATED-STORE
// DEPRECATED-STORE: error: unknown argument: '-analyzer-store=region'

// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-opt-analyze-nested-blocks %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPRECATED-NESTED-BLOCKS
// DEPRECATED-NESTED-BLOCKS: error: unknown argument: '-analyzer-opt-analyze-nested-blocks'

int empty(int x) {
  // CHECK: warning: Division by zero
  return x ? 0 : 0 / x;
}
