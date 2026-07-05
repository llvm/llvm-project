; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t
; RUN: mkdir -p %t/old/subdir
; RUN: opt -module-summary %s -o %t/old/subdir/1.obj
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t/old/subdir/2.obj
; RUN: opt -module-summary %p/Inputs/thinlto-empty.ll -o %t/old/3.obj

;; Ensure lld writes linked files to linked objects file.
; RUN: lld-link -entry:main -thinlto-index-only:%t/1.txt  %t/old/subdir/1.obj %t/old/subdir/2.obj %t/old/3.obj -out:%t/t.exe
; RUN: ls %t/old/subdir/1.obj.thinlto.bc
; RUN: ls %t/old/subdir/2.obj.thinlto.bc
; RUN: ls %t/old/3.obj.thinlto.bc
; RUN: FileCheck  --check-prefix=CHECK-NO-REPLACE %s < %t/1.txt
; CHECK-NO-REPLACE: old/subdir/1.obj
; CHECK-NO-REPLACE-NEXT: old/subdir/2.obj
; CHECK-NO-REPLACE-NEXT: old/3.obj

;; Check that this also works with thinlto-prefix-replace.
; RUN: lld-link -entry:main -thinlto-index-only:%t/2.txt -thinlto-prefix-replace:"%t/old/;%t/new/" %t/old/subdir/1.obj %t/old/subdir/2.obj %t/old/3.obj -out:%t/t.exe
; RUN: ls %t/new/subdir/1.obj.thinlto.bc
; RUN: ls %t/new/subdir/2.obj.thinlto.bc
; RUN: ls %t/new/3.obj.thinlto.bc
; RUN: FileCheck --check-prefix=CHECK-REPLACE-PREFIX  %s < %t/2.txt
; CHECK-REPLACE-PREFIX: new/subdir/1.obj
; CHECK-REPLACE-PREFIX-NEXT: new/subdir/2.obj
; CHECK-REPLACE-PREFIX-NEXT: new/3.obj

;; Check that this also works with replacing the prefix of linked objects.
; RUN: lld-link -entry:main -thinlto-index-only:%t/3.txt -thinlto-prefix-replace:"%t/old/;%t/new/;%t/obj/" %t/old/subdir/1.obj %t/old/subdir/2.obj %t/old/3.obj -out:%t/t.exe
; RUN: ls %t/new/subdir/1.obj.thinlto.bc
; RUN: ls %t/new/subdir/2.obj.thinlto.bc
; RUN: ls %t/new/3.obj.thinlto.bc
; RUN: FileCheck --check-prefix=CHECK-REPLACE-OBJECT-PREFIX %s < %t/3.txt
; CHECK-REPLACE-OBJECT-PREFIX: obj/subdir/1.obj
; CHECK-REPLACE-OBJECT-PREFIX-NEXT: obj/subdir/2.obj
; CHECK-REPLACE-OBJECT-PREFIX-NEXT: obj/3.obj

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

declare void @g(...)

define void @main() {
  call void (...) @g()
  ret void
}
