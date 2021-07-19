; REQUIRES: x86

;; Check that we internalize bitcode symbols (only) where possible, i.e. when
;; they are not referenced by undefined symbols originating from non-bitcode
;; files.

; RUN: rm -rf %t; split-file %s %t
; RUN: llvm-as %t/test.s -o %t/test.o
; RUN: llvm-as %t/baz.s -o %t/baz.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/regular.s -o %t/regular.o
; RUN: %lld -pie -lSystem %t/test.o %t/baz.o %t/regular.o -o %t/test -save-temps
; RUN: llvm-dis < %t/test.0.2.internalize.bc | FileCheck %s
; RUN: llvm-objdump --macho --syms %t/test | FileCheck %s --check-prefix=SYMTAB

;; Check that main is not internalized. This covers the case of bitcode symbols
;; referenced by undefined symbols that don't belong to any InputFile.
; CHECK: define void @main()

;; Check that the foo and bar functions are correctly internalized.
; CHECK: define internal void @bar()
; CHECK: define internal void @foo()

;; Check that a bitcode symbol that is referenced by a regular object file isn't
;; internalized.
; CHECK: define void @used_in_regular_obj()

;; Check that a bitcode symbol that is defined in another bitcode file gets
;; internalized.
; CHECK: define internal void @baz()

; Check foo and bar are not emitted to the .symtab
; SYMTAB-LABEL: SYMBOL TABLE:
; SYMTAB-NEXT:  g     F __TEXT,__text _main
; SYMTAB-NEXT:  g     F __TEXT,__text _used_in_regular_obj
; SYMTAB-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMTAB-NEXT:          *UND* dyld_stub_binder
; SYMTAB-EMPTY:

;--- test.s
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @baz()

define void @main() {
  call void @bar()
  call void @baz()
  ret void
}

define void @bar() {
  ret void
}

define hidden void @foo() {
  ret void
}

define void @used_in_regular_obj() {
  ret void
}

;--- baz.s
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @baz() {
  ret void
}

;--- regular.s
.data
.quad _used_in_regular_obj
