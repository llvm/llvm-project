; REQUIRES: x86

;; NOTE: We deviate significantly from ld64's behavior here. We treat common
;; bitcode symbols like regular common symbols, but ld64 gives them different
;; (and IMO very strange) precedence. This test documents the differences.

; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-summary %t/common.ll -o %t/common.o
; RUN: opt -module-summary %t/defined.ll -o %t/defined.o
; RUN: opt -module-summary %t/weak-defined.ll -o %t/weak-defined.o
; RUN: opt -module-summary %t/libfoo.ll -o %t/libfoo.o
; RUN: opt -module-summary %t/refs-foo.ll -o %t/refs-foo.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-defined.s -o %t/weak-defined-asm.o

; RUN: %lld -dylib -dylib %t/libfoo.o -o %t/libfoo.dylib

; RUN: llvm-ar rcs %t/defined.a %t/defined.o
; RUN: llvm-ar rcs %t/defined-and-common.a %t/defined.o %t/common.o
; RUN: llvm-ar rcs %t/common-and-defined.a %t/common.o %t/defined.o
; RUN: llvm-ar rcs %t/weak-defined-and-common.a %t/weak-defined.o %t/common.o
; RUN: llvm-ar rcs %t/common-and-weak-defined.a %t/common.o %t/weak-defined.o

;; Defined symbols take precedence over common bitcode symbols.
; RUN: %lld -dylib %t/defined.o %t/common.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED
; RUN: %lld -dylib %t/common.o %t/defined.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED

;; Defined symbols have the same precedence as common bitcode symbols within
;; an archive.
; RUN: %lld -dylib %t/defined-and-common.a %t/refs-foo.o -o %t/refs-foo
; RUN: llvm-objdump --syms %t/refs-foo | FileCheck %s --check-prefix=DEFINED
; RUN: %lld -dylib %t/common-and-defined.a %t/refs-foo.o -o %t/refs-foo
; RUN: llvm-objdump --syms %t/refs-foo | FileCheck %s --check-prefix=COMMON

;; ld64: Weak bitcode symbols have the same precedence as common bitcode symbols.
;; lld: Weak bitcode symbols take precedence over common bitcode symbols.
; RUN: %lld -dylib %t/weak-defined.o %t/common.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED
; RUN: %lld -dylib %t/common.o %t/weak-defined.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED

;; Weak non-bitcode symbols take precedence over common bitcode symbols.
; RUN: %lld -dylib %t/weak-defined-asm.o %t/common.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED
; RUN: %lld -dylib %t/common.o %t/weak-defined-asm.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=WEAK-DEFINED

;; ld64: Archive symbols take precedence over common bitcode symbols.
;; lld: Common bitcode symbols take precedence over archive symbols.
; RUN: %lld -dylib %t/defined.a %t/common.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=COMMON
; COM (ld64): llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED
; RUN: %lld -dylib %t/common.o %t/defined.a -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=COMMON
; COM (ld64): llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DEFINED

;; ld64: Dylib symbols take precedence over common bitcode symbols.
;; lld: Common bitcode symbols take precedence over dylib symbols.
; RUN: %lld -dylib %t/libfoo.dylib %t/common.o %t/refs-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=COMMON
; COM (ld64): llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DYLIB
; RUN: %lld -dylib %t/common.o %t/libfoo.dylib %t/refs-foo.o -o %t/test
; RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=COMMON
; COM (ld64): llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DYLIB

; COMMON:       g     O __DATA,__common _foo
; DEFINED:      g     O __DATA,__data _foo
; WEAK-DEFINED: w     O __DATA,__data _foo
; DYLIB:        *UND* _foo

;--- common.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@foo = common global i8 0

;--- defined.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@foo = global i8 12

;--- weak-defined.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@foo = weak global i8 12

;--- weak-defined.s
.globl _foo
.weak_definition _foo
.data
_foo:

;--- libfoo.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@foo = common global i8 0

;--- refs-foo.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@foo = external global i8

define void @f() {
  %1 = load i8, i8* @foo
  ret void
}
