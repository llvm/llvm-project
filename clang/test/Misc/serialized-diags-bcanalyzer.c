// RUN: %clang -Wall -fsyntax-only %s --serialize-diagnostics %t.diag > /dev/null 2>&1
// RUN: llvm-bcanalyzer -dump %t.diag | FileCheck %s
// CHECK: Stream type: Clang Serialized Diagnostics
