; REQUIRES: x86

; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: opt -module-summary %s -o 1.bc
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o 2.bc

; RUN: ld.lld --plugin-opt=obj-path=4.o -shared 1.bc 2.bc -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM3
; RUN: llvm-objdump -d 4.o1 | FileCheck %s --check-prefix=CHECK1
; RUN: llvm-objdump -d 4.o2 | FileCheck %s --check-prefix=CHECK2

; NM3:      T f
; NM3-NEXT: T g

; CHECK1:       file format elf64-x86-64
; CHECK1-EMPTY:
; CHECK1-NEXT:  Disassembly of section .text.f:
; CHECK1-EMPTY:
; CHECK1-NEXT:  <f>:
; CHECK1-NEXT:    retq
; CHECK1-NOT:   {{.}}

; CHECK2:       file format elf64-x86-64
; CHECK2-EMPTY:
; CHECK2-NEXT:  Disassembly of section .text.g:
; CHECK2-EMPTY:
; CHECK2-NEXT:  <g>:
; CHECK2-NEXT:    retq
; CHECK2-NOT:   {{.}}

;; With --thinlto-index-only, --lto-obj-path= creates just one file.
; RUN: rm -f 4.o 4.o1 4.o2
; RUN: ld.lld --thinlto-index-only --lto-obj-path=4.o -shared 1.bc 2.bc -o /dev/null
; RUN: llvm-objdump -d 4.o | FileCheck %s --check-prefix=EMPTY
; RUN: not ls 4.o1
; RUN: not ls 4.o2

;; Test --plugin-opt=obj-path=.
; RUN: rm -f 4.o
; RUN: ld.lld --plugin-opt=thinlto-index-only --plugin-opt=obj-path=4.o -shared 1.bc 2.bc -o /dev/null
; RUN: llvm-objdump -d 4.o | FileCheck %s --check-prefix=EMPTY

;; Ensure lld emits empty combined module if specific obj-path.
; RUN: rm -fr objpath && mkdir objpath
; RUN: ld.lld --plugin-opt=obj-path=4.o -shared 1.bc 2.bc -o objpath/a.out --save-temps
; RUN: ls objpath/a.out*.lto.* | count 3

;; Ensure lld does not emit empty combined module in default.
; RUN: rm -fr objpath && mkdir objpath
; RUN: ld.lld -shared 1.bc 2.bc -o objpath/a.out --save-temps
; RUN: ls objpath/a.out*.lto.* | count 2

; EMPTY:     file format elf64-x86-64
; EMPTY-NOT: {{.}}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
