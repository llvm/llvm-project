// RUN: %clang_analyze_cc1 -analyzer-checker=core %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK,DEPRECATED-STORE
// DEPRECATED-STORE: warning: analyzer option '-analyzer-store' is deprecated. This flag will be removed in clang-16, and passing this option will be an error.

// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-opt-analyze-nested-blocks %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK,DEPRECATED-NESTED-BLOCKS
// DEPRECATED-NESTED-BLOCKS: warning: analyzer option '-analyzer-opt-analyze-nested-blocks' is deprecated. This flag will be removed in clang-16, and passing this option will be an error.

// RUN: %clang_analyze_cc1 -analyzer-checker=core %s --help 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-HELP
// CHECK-HELP: Analyze the definitions of blocks in addition to functions [DEPRECATED, removing in clang-16]
// CHECK-HELP: -analyzer-store <value> Source Code Analysis - Abstract Memory Store Models [DEPRECATED, removing in clang-16]

int empty(int x) {
  // CHECK: warning: Division by zero
  return x ? 0 : 0 / x;
}
