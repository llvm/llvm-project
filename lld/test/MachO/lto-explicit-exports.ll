; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

;; Check that `-exported_symbol` causes all non-exported symbols to be marked
;; as hidden before LTO. We don't want to downgrade them to private extern only
;; after LTO runs as that likely causes LTO to miss optimization opportunities.

; RUN: llvm-as %t/foo.ll -o %t/foo.o
; RUN: llvm-as %t/refs-foo.ll -o %t/refs-foo.o

; RUN: %lld -lSystem -dylib %t/foo.o %t/refs-foo.o -o %t/test-fulllto \
; RUN:  -save-temps -exported_symbol _refs_foo -exported_symbol _same_module_caller

; RUN: llvm-dis %t/test-fulllto.0.2.internalize.bc -o - | FileCheck %s --check-prefix=FULLLTO
; RUN: llvm-objdump --macho --syms %t/test-fulllto | FileCheck %s --check-prefix=FULLLTO-SYMS

; FULLLTO: define internal void @foo()
; FULLLTO: define internal void @same_module_callee()
; FULLLTO: define dso_local void @same_module_caller()
; FULLLTO: define dso_local void @refs_foo()

;; LTO is able to elide the hidden symbols, and they will be entirely absent
;; from the final symbol table.

; FULLLTO-SYMS:       SYMBOL TABLE:
; FULLLTO-SYMS:       g     F __TEXT,__text _refs_foo
; FULLLTO-SYMS:       g     F __TEXT,__text _same_module_caller
; FULLLTO-SYMS:       *UND* dyld_stub_binder
; FULLLTO-SYMS-EMPTY:

;; ThinLTO is unable to internalize symbols that are referenced from another
;; module. Verify that we still mark the final symbol as private extern.

; RUN: opt -module-summary %t/foo.ll -o %t/foo.thinlto.o
; RUN: opt -module-summary %t/refs-foo.ll -o %t/refs-foo.thinlto.o

; RUN: %lld -lSystem -dylib %t/foo.thinlto.o %t/refs-foo.thinlto.o -o %t/test-thinlto \
; RUN:  -save-temps -exported_symbol _refs_foo -exported_symbol _same_module_caller

; RUN: llvm-dis %t/foo.thinlto.o.2.internalize.bc -o - | FileCheck %s --check-prefix=THINLTO-FOO
; RUN: llvm-dis %t/refs-foo.thinlto.o.2.internalize.bc -o - | FileCheck %s --check-prefix=THINLTO-REFS-FOO
; RUN: llvm-objdump --macho --syms %t/test-thinlto | FileCheck %s --check-prefix=THINLTO-SYMS

; THINLTO-FOO: define dso_local void @foo()
; THINLTO-FOO: define internal void @same_module_callee()
; THINLTO-REFS-FOO: declare dso_local void @foo()
; THINLTO-REFS-FOO: define dso_local void @refs_foo()

; THINLTO-SYMS: l     F __TEXT,__text .hidden _foo
; THINLTO-SYMS: g     F __TEXT,__text _refs_foo
; THINLTO-SYMS: g     F __TEXT,__text _same_module_caller

;--- foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

define void @same_module_callee() {
  ret void
}

define void @same_module_caller() {
  call void @same_module_callee()
  ret void
}

;--- refs-foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @foo()

define void @refs_foo() {
  call void @foo()
  ret void
}
