// RUN: not clang-tidy -checks=-*,llvm-namespace-comment %s -- -serialize-diagnostics %t | FileCheck %s
// CHECK: :[[@LINE+1]]:12: error: expected ';' after struct [clang-diagnostic-error]
struct A {}
