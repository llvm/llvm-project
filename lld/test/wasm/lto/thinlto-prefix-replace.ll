; Copied from ELF/lto/thinlto-prefix-replace.ll
; Check that changing the output path via thinlto-prefix-replace works
; RUN: mkdir -p %t/oldpath
; RUN: opt -module-summary %s -o %t/oldpath/thinlto_prefix_replace.o

; Ensure that there is no existing file at the new path, so we properly
; test the creation of the new file there.
; RUN: rm -f %t/newpath/thinlto_prefix_replace.o.thinlto.bc
; RUN: wasm-ld --thinlto-index-only --thinlto-prefix-replace="%t/oldpath/;%t/newpath/" -shared %t/oldpath/thinlto_prefix_replace.o -o %t/thinlto_prefix_replace
; RUN: ls %t/newpath/thinlto_prefix_replace.o.thinlto.bc

; Ensure that lld generates error if prefix replace option does not have 'old;new' format.
; RUN: rm -f %t/newpath/thinlto_prefix_replace.o.thinlto.bc
; RUN: not wasm-ld --thinlto-index-only --thinlto-prefix-replace=abc:def -shared %t/oldpath/thinlto_prefix_replace.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
; ERR: --thinlto-prefix-replace= expects 'old;new' format, but got abc:def

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define void @f() {
entry:
  ret void
}
