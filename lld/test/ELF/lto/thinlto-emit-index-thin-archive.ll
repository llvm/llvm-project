; REQUIRES: x86

; RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir
; RUN: mkdir dir1 dir2
; RUN: llvm-as %s -o ./dir1/main.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o ./dir1/unused.o
; RUN: llvm-as %p/Inputs/thin1.ll -o ./dir1/thin.o
; RUN: llvm-as %p/Inputs/thin2.ll -o ./dir2/thin.o
; RUN: llvm-ar crT ./dir2/lib.a dir1/unused.o dir1/thin.o dir2/thin.o

;; For a thin archive referencing object files in a different directory,
;; emit index files (lib.a($member at $offset).thinlto.bc) in the directory
;; containing the archive, even in the lazy case. The information about the
;; referenced member's directory is lost.
; RUN: ld.lld --thinlto-emit-index-files ./dir2/lib.a ./dir1/main.o -o c --save-temps
; RUN: ls ./dir2 | FileCheck %s --check-prefix CHECK-UNUSED

; CHECK-UNUSED: lib.a(unused.o at {{[1-9][0-9]+}})

;; Index files emitted from object files in a thin archive should have the
;; offset in the archive specified to avoid collisions
; RUN: FileCheck %s < c.resolution.txt --check-prefix CHECK-COLLISION

; CHECK-COLLISION: dir1/main.o
; CHECK-COLLISION: dir2/lib.a(thin.o at {{[1-9][0-9]+}})
; CHECK-COLLISION-NEXT: -r=./dir2/lib.a(thin.o at {{[1-9][0-9]+}}),blah,pl
; CHECK-COLLISION: dir2/lib.a(thin.o at {{[1-9][0-9]+}})
; CHECK-COLLISION-NEXT: -r=./dir2/lib.a(thin.o at {{[1-9][0-9]+}}),foo,pl

;; Clean up
; RUN: rm -rf ./dir1/*.thinlto.bc
; RUN: rm -rf ./dir2/*.thinlto.bc
;; Empty index files for unused files in thin archives should still be emitted
;; in the same format when using --whole-archive
; RUN: ld.lld --thinlto-emit-index-files --whole-archive ./dir2/lib.a ./dir1/main.o -o d
; RUN: ls ./dir2 | FileCheck %s --check-prefix CHECK-UNUSED

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @blah(i32 %meh)
declare i32 @foo(i32 %goo)

define void @_start() {
entry:
  call i32 @foo(i32 0)
  call i32 @blah(i32 0)
  ret void
}
