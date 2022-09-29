// RUN: not %clang_cc1 -mrelocation-model tinkywinky \
// RUN: -emit-llvm %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

// CHECK-INVALID: error: invalid value 'tinkywinky' in '-mrelocation-model tinkywinky'
