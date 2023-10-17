; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
; RUN: mkdir %t/a %t/b
; RUN: opt -thinlto-bc -o %t/main.o %t/main.ll
; RUN: opt -thinlto-bc -o %t/a/bar.o %t/foo.ll
; RUN: opt -thinlto-bc -o %t/b/bar.o %t/bar.ll
; RUN: llvm-ar crs %t/libbar.a %t/a/bar.o %t/b/bar.o
; RUN: %lld -save-temps %t/main.o %t/libbar.a -o %t/test
; RUN: FileCheck %s --check-prefix=SAME-ARCHIVE < %t/test.resolution.txt

; RUN: llvm-ar crs %t/liba.a %t/a/bar.o
; RUN: llvm-ar crs %t/libb.a %t/b/bar.o
; RUN: %lld -save-temps %t/main.o %t/liba.a %t/libb.a -o %t/test
; RUN: FileCheck %s --check-prefix=DIFFERENT-ARCHIVES < %t/test.resolution.txt

; SAME-ARCHIVE: libbar.a(bar.o)[[#OFFSET:]]
; SAME-ARCHIVE-NEXT: -r={{.*}}/libbar.a(bar.o)[[#OFFSET:]],_foo,p
; SAME-ARCHIVE-NEXT: libbar.a(bar.o)[[#OTHEROFFSET:]]
; SAME-ARCHIVE-NEXT: -r={{.*}}/libbar.a(bar.o)[[#OTHEROFFSET:]],_bar,p

; DIFFERENT-ARCHIVES: liba.a(bar.o)[[#OFFSET:]]
; DIFFERENT-ARCHIVES-NEXT: -r={{.*}}/liba.a(bar.o)[[#OFFSET:]],_foo,p
; DIFFERENT-ARCHIVES-NEXT: libb.a(bar.o)[[#OTHEROFFSET:]]
; DIFFERENT-ARCHIVES-NEXT: -r={{.*}}/libb.a(bar.o)[[#OTHEROFFSET:]],_bar,p

;--- main.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @bar()
declare void @foo()

define i32 @main() {
  call void @foo()
  call void @bar()
  ret i32 0
}

;--- foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

;--- bar.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @bar() {
  ret void
}
