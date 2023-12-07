; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: mkdir -p old/subdir
; RUN: opt -module-summary %s -o old/subdir/1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o old/subdir/2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o old/3.o

;; Ensure lld writes linked files to linked objects file.
; RUN: ld.lld --thinlto-index-only=1.txt -shared old/subdir/1.o old/subdir/2.o old/3.o -o /dev/null
; RUN: ls old/subdir/1.o.thinlto.bc
; RUN: ls old/subdir/2.o.thinlto.bc
; RUN: ls old/3.o.thinlto.bc
; RUN: FileCheck --match-full-lines --check-prefix=CHECK-NO-REPLACE %s < 1.txt
; CHECK-NO-REPLACE: old/subdir/1.o
; CHECK-NO-REPLACE-NEXT: old/subdir/2.o
; CHECK-NO-REPLACE-NEXT: old/3.o

;; Check that this also works with thinlto-prefix-replace.
; RUN: ld.lld --thinlto-index-only=2.txt --thinlto-prefix-replace="old/;new/" -shared old/subdir/1.o old/subdir/2.o old/3.o -o /dev/null
; RUN: ls new/subdir/1.o.thinlto.bc
; RUN: ls new/subdir/2.o.thinlto.bc
; RUN: ls new/3.o.thinlto.bc
; RUN: FileCheck --match-full-lines --check-prefix=CHECK-REPLACE-PREFIX  %s < 2.txt
; CHECK-REPLACE-PREFIX: new/subdir/1.o
; CHECK-REPLACE-PREFIX-NEXT: new/subdir/2.o
; CHECK-REPLACE-PREFIX-NEXT: new/3.o

;; Check that this also works with replacing the prefix of linked objects.
; RUN: ld.lld --thinlto-index-only=3.txt --thinlto-prefix-replace="old/;new/;obj/" -shared old/subdir/1.o old/subdir/2.o old/3.o -o /dev/null
; RUN: ls new/subdir/1.o.thinlto.bc
; RUN: ls new/subdir/2.o.thinlto.bc
; RUN: ls new/3.o.thinlto.bc
; RUN: FileCheck --match-full-lines --check-prefix=CHECK-REPLACE-OBJECT-PREFIX %s < 3.txt
; CHECK-REPLACE-OBJECT-PREFIX: obj/subdir/1.o
; CHECK-REPLACE-OBJECT-PREFIX-NEXT: obj/subdir/2.o
; CHECK-REPLACE-OBJECT-PREFIX-NEXT: obj/3.o

; Create an error if prefix replace option have 'old;new;obj' format but index file is not set. Ensure that the error is about thinlto-prefix-replace.
; RUN: not ld.lld --thinlto-prefix-replace="old/;new/;obj/" -shared old/subdir/1.o old/subdir/2.o old/3.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: error: --thinlto-prefix-replace=old_dir;new_dir;obj_dir must be used with --thinlto-index-only=

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
