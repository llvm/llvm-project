; RUN: llvm-as %s -o %t1.o
; RUN: ld.lld -m elf_x86_64 %t1.o -o %t -shared -save-temps
; RUN: llvm-dis < %t.lto.bc | FileCheck %s
; RUN: llvm-readobj -t %t | FileCheck %s --check-prefix=SHARED

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i8 0, align 8

; Shared library case, we ensure that the bitcode generated file
; has not the a symbol but is appears in the final shared library
; produced.
; CHECK-NOT: @a = common global i8 0, align 8

; SHARED: Symbol {
; SHARED:   Name: a
; SHARED-NEXT:   Value: 0x2000
; SHARED-NEXT:   Size: 1
; SHARED-NEXT:   Binding: Global
; SHARED-NEXT:   Type: None
; SHARED-NEXT:   Other: 0
; SHARED-NEXT:   Section: .bss
; SHARED-NEXT: }
