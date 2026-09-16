// RUN: not %clang_analyze_cc1 -analyzer-constraints=unsupported-z3 %s 2>&1 | FileCheck %s
// UNSUPPORTED: z3

// CHECK: error: analyzer constraint manager 'unsupported-z3' is only available
// CHECK-SAME: if LLVM was built with -DLLVM_ENABLE_Z3_SOLVER=ON
