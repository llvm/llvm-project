; REQUIRES: x86

; RUN: rm -rf %t; split-file %s %t

; RUN: opt -module-summary %t/defined.ll -o %t/defined.o
; RUN: opt -module-summary %t/weak-defined.ll -o %t/weak-defined.o
; RUN: opt -module-summary %t/archive.ll -o %t/archive.o
; RUN: opt -module-summary %t/calls-foo.ll -o %t/calls-foo.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-defined.s -o %t/weak-defined-asm.o

; RUN: %lld -lSystem -dylib %t/defined.o -o %t/libfoo.dylib
; RUN: %lld -lSystem -dylib %t/weak-defined.o -o %t/libweakfoo.dylib

; RUN: llvm-ar rcs %t/archive.a %t/archive.o

;; Regular defined symbols take precedence over weak ones.
; RUN: %lld -lSystem %t/defined.o %t/weak-defined.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED
; RUN: %lld -lSystem %t/weak-defined.o %t/defined.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED

;; Regular defined symbols take precedence over weak non-bitcode ones.
; RUN: %lld -lSystem %t/defined.o %t/weak-defined-asm.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED
; RUN: %lld -lSystem %t/weak-defined-asm.o %t/defined.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED

;; NOTE: we are deviating from ld64's behavior here.
;; ld64: Weak non-bitcode symbols take precedence over weak bitcode ones.
;; lld: Weak non-bitcode symbols have the same precedence as weak bitcode ones.
; RUN: %lld -lSystem %t/weak-defined.o %t/weak-defined-asm.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED
; COM (ld64): llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED-ASM
; RUN: %lld -lSystem %t/weak-defined-asm.o %t/weak-defined.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED-ASM

;; Weak defined symbols take precedence over dylib symbols.
; RUN: %lld -lSystem %t/weak-defined.o %t/libfoo.dylib %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED
; RUN: %lld -lSystem %t/libfoo.dylib %t/weak-defined.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED

;; Weak defined symbols take precedence over archive symbols.
; RUN: %lld -lSystem %t/archive.a %t/weak-defined.o %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED
; RUN: %lld -lSystem %t/weak-defined.o %t/archive.a %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED

;; Archive symbols have the same precedence as dylib symbols.
; RUN: %lld -lSystem %t/archive.a %t/libfoo.dylib %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=ARCHIVE
; RUN: %lld -lSystem %t/libfoo.dylib %t/archive.a %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DYLIB

;; Archive symbols take precedence over weak dylib symbols.
; RUN: %lld -lSystem %t/archive.a %t/libweakfoo.dylib %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=ARCHIVE
; RUN: %lld -lSystem %t/libweakfoo.dylib %t/archive.a %t/calls-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=ARCHIVE

; DEFINED:          g     O __TEXT,defined _foo
; WEAK-DEFINED:     w     O __TEXT,weak_defined _foo
; WEAK-DEFINED-ASM: w     O __TEXT,weak_defined_asm _foo
; ARCHIVE:          g     O __TEXT,archive _foo
; DYLIB:            *UND* _foo

;--- defined.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define void @foo() section "__TEXT,defined" {
  ret void
}

;--- weak-defined.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define weak void @foo() section "__TEXT,weak_defined" {
  ret void
}

;--- weak-defined.s
.globl _foo
.weak_definition _foo
.section __TEXT,weak_defined_asm
_foo:

;--- archive.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define void @foo() section "__TEXT,archive" {
  ret void
}

;--- calls-foo.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

declare void @foo()

define void @main() {
  call void @foo()
  ret void
}
