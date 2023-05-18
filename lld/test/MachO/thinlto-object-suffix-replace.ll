; REQUIRES: x86
;; Test to make sure the thinlto-object-suffix-replace option is handled
;; correctly.
; RUN: rm -rf %t && mkdir %t && cd %t

;; Generate bitcode file with summary, as well as a minimized bitcode without
; the debug metadata for the thin link.
; RUN: opt --thinlto-bc %s -thin-link-bitcode-file=1.thinlink.bc -o 1.o

;; First perform the thin link on the normal bitcode file, and save the
;; resulting index.
; RUN: %lld --thinlto-index-only -dylib 1.o -o 3
; RUN: cp 1.o.thinlto.bc 1.o.thinlto.bc.orig

;; Next perform the thin link on the minimized bitcode file, and compare dump
;; of the resulting index to the above dump to ensure they are identical.
; RUN: rm -f 1.o.thinlto.bc
;; Make sure it isn't inadvertently using the regular bitcode file.
; RUN: rm -f 1.o
; RUN: %lld --thinlto-index-only --thinlto-object-suffix-replace=".thinlink.bc;.o" \
; RUN:   -dylib 1.thinlink.bc -o 3
; RUN: cmp 1.o.thinlto.bc.orig 1.o.thinlto.bc

;; Ensure lld generates error if object suffix replace option does not have 'old;new' format
; RUN: rm -f 1.o.thinlto.bc
; RUN: not %lld --thinlto-index-only --thinlto-object-suffix-replace="abc:def" -dylib 1.thinlink.bc \
; RUN:   -o 3 2>&1 | FileCheck %s --check-prefix=ERR1
; ERR1: --thinlto-object-suffix-replace= expects 'old;new' format, but got abc:def

;; If filename does not end with old suffix, no suffix change should occur,
;; so ".thinlto.bc" will simply be appended to the input file name.
; RUN: rm -f 1.thinlink.bc.thinlto.bc
; RUN: %lld --thinlto-index-only --thinlto-object-suffix-replace=".abc;.o" -dylib 1.thinlink.bc -o /dev/null
; RUN: ls 1.thinlink.bc.thinlto.bc

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @f() {
entry:
  ret void
}

!llvm.dbg.cu = !{}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!1}
