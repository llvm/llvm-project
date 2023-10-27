; RUN: %clang -fbounds-safety-experimental -x ir -S %s -o /dev/null 2>&1 | FileCheck %s
; RUN: %clang_cc1 -fbounds-safety-experimental -x ir -S %s -o /dev/null 2>&1 | FileCheck %s

; CHECK-NOT: warning: '-fbounds-safety' is ignored for LLVM IR
; CHECK-NOT: error: bounds safety is only supported for C