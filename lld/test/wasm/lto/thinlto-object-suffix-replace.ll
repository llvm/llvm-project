;; Copied from ELF/lto/thinlto-object-suffix-replace.ll
;; Test to make sure the thinlto-object-suffix-replace option is handled
;; correctly.
; RUN: rm -rf %t && mkdir %t && cd %t

;; Generate bitcode file with summary, as well as a minimized bitcode without
; the debug metadata for the thin link.
; RUN: opt --thinlto-bc %s -thin-link-bitcode-file=1.thinlink.bc -o 1.o

;; First perform the thin link on the normal bitcode file, and save the
;; resulting index.
; RUN: wasm-ld --thinlto-index-only -shared 1.o -o 3
; RUN: cp 1.o.thinlto.bc 1.o.thinlto.bc.orig

;; Next perform the thin link on the minimized bitcode file, and compare dump
;; of the resulting index to the above dump to ensure they are identical.
; RUN: rm -f 1.o.thinlto.bc
;; Make sure it isn't inadvertently using the regular bitcode file.
; RUN: rm -f 1.o
; RUN: wasm-ld --thinlto-index-only --thinlto-object-suffix-replace=".thinlink.bc;.o" \
; RUN:   -shared 1.thinlink.bc -o 3
; RUN: cmp 1.o.thinlto.bc.orig 1.o.thinlto.bc

;; Ensure lld generates error if object suffix replace option does not have 'old;new' format
; RUN: rm -f 1.o.thinlto.bc
; RUN: not wasm-ld --thinlto-index-only --thinlto-object-suffix-replace="abc:def" -shared 1.thinlink.bc \
; RUN:   -o 3 2>&1 | FileCheck %s --check-prefix=ERR1
; ERR1: --thinlto-object-suffix-replace= expects 'old;new' format, but got abc:def

;; If filename does not end with old suffix, no suffix change should occur,
;; so ".thinlto.bc" will simply be appended to the input file name.
; RUN: rm -f 1.thinlink.bc.thinlto.bc
; RUN: wasm-ld --thinlto-index-only --thinlto-object-suffix-replace=".abc;.o" -shared 1.thinlink.bc -o /dev/null
; RUN: ls 1.thinlink.bc.thinlto.bc

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define void @f() {
entry:
  ret void
}

!llvm.dbg.cu = !{}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!1}
