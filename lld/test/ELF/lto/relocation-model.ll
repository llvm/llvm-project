; REQUIRES: x86

; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/relocation-model-pic.ll -o %t.pic.o

;; Non-PIC source.

; RUN: ld.lld %t.o -o %t-out -save-temps -shared
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: ld.lld %t.o -o %t-out -save-temps --export-dynamic --noinhibit-exec -pie
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: ld.lld %t.o -o %t-out -save-temps --export-dynamic --noinhibit-exec
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=STATIC


;; PIC source.

; RUN: ld.lld %t.pic.o -o %t-out -save-temps -shared
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: ld.lld %t.pic.o -o %t-out -save-temps --export-dynamic --noinhibit-exec -pie
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: ld.lld %t.pic.o -o %t-out -save-temps --export-dynamic --noinhibit-exec
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=STATIC


;; Explicit flag.

; RUN: ld.lld %t.o -o %t-out -save-temps -r -mllvm -relocation-model=pic
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=PIC

; RUN: ld.lld %t.o -o %t-out -save-temps -r -mllvm -relocation-model=static
; RUN: llvm-readobj -r %t-out.lto.o | FileCheck %s --check-prefix=STATIC


; PIC: R_X86_64_REX_GOTPCRELX foo
; STATIC: R_X86_64_PC32 foo

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external global i32
define i32 @main() {
  %t = load i32, i32* @foo
  ret i32 %t
}
