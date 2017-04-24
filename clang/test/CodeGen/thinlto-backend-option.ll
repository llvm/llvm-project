; Test to ensure -backend-options work when invoking the ThinLTO backend path.

; This test uses a non-existent backend option to test that backend options are
; being parsed. While it's more important that the existing options are parsed
; than that this error is produced, this provides a reliable way to test this
; scenario independent of any particular backend options that may exist now or
; in the future.

; RUN: %clang -flto=thin -c -o %t.o %s
; RUN: llvm-lto -thinlto -o %t %t.o
; RUN: not %clang_cc1 -x ir %t.o -fthinlto-index=%t.thinlto.bc -backend-option -nonexistent -emit-obj -o /dev/null 2>&1 | FileCheck %s

; CHECK: clang: Unknown command line argument '-nonexistent'
