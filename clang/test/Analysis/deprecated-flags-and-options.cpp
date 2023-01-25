// RUN: %clang_analyze_cc1 -analyzer-checker=core %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK

// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPRECATED-STORE
// DEPRECATED-STORE: error: unknown argument: '-analyzer-store=region'

// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-opt-analyze-nested-blocks %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPRECATED-NESTED-BLOCKS
// DEPRECATED-NESTED-BLOCKS: error: unknown argument: '-analyzer-opt-analyze-nested-blocks'

// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config consider-single-element-arrays-as-flexible-array-members=true %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK,DEPRECATED-SINGLE-ELEM-FAM
// DEPRECATED-SINGLE-ELEM-FAM: warning: analyzer option 'consider-single-element-arrays-as-flexible-array-members' is deprecated. This flag will be removed in clang-17, and passing this option will be an error. Use '-fstrict-flex-arrays=<N>' instead.

// RUN: %clang_analyze_cc1 -analyzer-config-help 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK-HELP
// CHECK-HELP:      [DEPRECATED, removing in clang-17; use '-fstrict-flex-arrays=<N>'
// CHECK-HELP-NEXT: instead] (default: true)

int empty(int x) {
  // CHECK: warning: Division by zero
  return x ? 0 : 0 / x;
}
