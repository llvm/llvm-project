; REQUIRES: x86

;; Check that changing the output path via thinlto-prefix-replace works
; RUN: mkdir -p %t/oldpath
; RUN: opt -module-summary %s -o %t/oldpath/thinlto_prefix_replace.o

;; Ensure that there is no existing file at the new path, so we properly
;; test the creation of the new file there.
; RUN: rm -f %t/newpath/thinlto_prefix_replace.o.thinlto.bc
; RUN: %lld --thinlto-index-only --thinlto-prefix-replace="%t/oldpath/;%t/newpath/" -dylib %t/oldpath/thinlto_prefix_replace.o -o %t/thinlto_prefix_replace
; RUN: ls %t/newpath/thinlto_prefix_replace.o.thinlto.bc

;; Ensure that lld generates error if prefix replace option does not have 'old;new' format.
; RUN: rm -f %t/newpath/thinlto_prefix_replace.o.thinlto.bc
; RUN: not %lld --thinlto-index-only --thinlto-prefix-replace=abc:def -dylib %t/oldpath/thinlto_prefix_replace.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
; ERR: --thinlto-prefix-replace= expects 'old;new' format, but got abc:def

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @f() {
entry:
  ret void
}
