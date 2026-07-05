; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t
; RUN: mkdir -p %t/old/subdir

; RUN: opt -module-summary %t/f.ll -o %t/old/subdir/1.o
; RUN: opt -module-summary %t/g.ll -o %t/old/subdir/2.o
; RUN: opt -module-summary %t/empty.ll -o %t/old/3.o

;; Ensure lld writes linked files to linked objects file.
; RUN: %lld --thinlto-index-only=%t/1.txt -dylib %t/old/subdir/1.o %t/old/subdir/2.o %t/old/3.o -o /dev/null
; RUN: ls %t/old/subdir/1.o.thinlto.bc
; RUN: ls %t/old/subdir/2.o.thinlto.bc
; RUN: ls %t/old/3.o.thinlto.bc
; RUN: FileCheck --check-prefix=CHECK-NO-REPLACE %s < %t/1.txt
; CHECK-NO-REPLACE: old/subdir/1.o
; CHECK-NO-REPLACE-NEXT: old/subdir/2.o
; CHECK-NO-REPLACE-NEXT: old/3.o

;; Check that this also works with thinlto-prefix-replace.
; RUN: %lld --thinlto-index-only=%t/2.txt --thinlto-prefix-replace="%t/old/;%t/new/" -dylib %t/old/subdir/1.o %t/old/subdir/2.o %t/old/3.o -o /dev/null
; RUN: ls %t/new/subdir/1.o.thinlto.bc
; RUN: ls %t/new/subdir/2.o.thinlto.bc
; RUN: ls %t/new/3.o.thinlto.bc
; RUN: FileCheck --check-prefix=CHECK-REPLACE-PREFIX  %s < %t/2.txt
; CHECK-REPLACE-PREFIX: new/subdir/1.o
; CHECK-REPLACE-PREFIX-NEXT: new/subdir/2.o
; CHECK-REPLACE-PREFIX-NEXT: new/3.o


;; Check that this also works with replacing the prefix of linked objects.
; RUN: %lld --thinlto-index-only=%t/3.txt --thinlto-prefix-replace="%t/old/;%t/new/;%t/obj/" -dylib %t/old/subdir/1.o %t/old/subdir/2.o %t/old/3.o -o /dev/null
; RUN: ls %t/new/subdir/1.o.thinlto.bc
; RUN: ls %t/new/subdir/2.o.thinlto.bc
; RUN: ls %t/new/3.o.thinlto.bc
; RUN: FileCheck --check-prefix=CHECK-REPLACE-OBJECT-PREFIX %s < %t/3.txt
; CHECK-REPLACE-OBJECT-PREFIX: obj/subdir/1.o
; CHECK-REPLACE-OBJECT-PREFIX-NEXT: obj/subdir/2.o
; CHECK-REPLACE-OBJECT-PREFIX-NEXT: obj/3.o

; Create an error if prefix replace option have 'old;new;obj' format but index file is not set. Ensure that the error is about thinlto-prefix-replace.
; RUN: not %lld --thinlto-prefix-replace="%t/old/;%t/new/;%t/obj/" -dylib %t/old/subdir/1.o %t/old/subdir/2.o %t/old/3.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: error: --thinlto-prefix-replace=old_dir;new_dir;obj_dir must be used with --thinlto-index-only=

;--- f.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- g.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @g() {
entry:
  ret void
}

;--- empty.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"
