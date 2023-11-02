; RUN: %clang -fexperimental-bounds-safety -c -x ir -S %s -o /dev/null 2>&1 | FileCheck %s

; The option is silently ignored for LLVM IR. This conforms to the behavior of most other cflags.
; CHECK-NOT: warning: warning: argument unused during compilation: '-fexperimental-bounds-safety'
; CHECK-NOT: error: bounds safety is only supported for C