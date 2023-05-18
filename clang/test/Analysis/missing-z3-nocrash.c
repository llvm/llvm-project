// RUN: not %clang_analyze_cc1 -analyzer-constraints=z3 %s 2>&1 | FileCheck %s
// REQUIRES: no-z3

// CHECK: error: analyzer constraint manager 'z3' is only available if LLVM
// CHECK: was built with -DLLVM_ENABLE_Z3_SOLVER=ON
