; Test to ensure -mllvm work when invoking the ThinLTO backend path.

; This test uses a non-existent backend option to test that backend options are
; being parsed. While it's more important that the existing options are parsed
; than that this error is produced, this provides a reliable way to test this
; scenario independent of any particular backend options that may exist now or
; in the future.

; XFAIL: target={{.*}}-aix{{.*}}

; RUN: %clang -flto=thin -c -o %t.o %s
; RUN: llvm-lto -thinlto -o %t %t.o
; RUN: not %clang_cc1 -x ir %t.o -fthinlto-index=%t.thinlto.bc -mllvm -nonexistent -emit-obj -o /dev/null 2>&1 | FileCheck %s -check-prefix=UNKNOWN
; UNKNOWN: clang (LLVM option parsing): Unknown command line argument '-nonexistent'

; RUN: not %clang_cc1 -flto=thinfoo 2>&1 | FileCheck %s -check-prefix=INVALID
; INVALID: error: invalid value 'thinfoo' in '-flto=thinfoo'
