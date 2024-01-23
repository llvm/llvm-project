; REQUIRES: x86
;; Test --lto-obj-path= for regular LTO.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: mkdir d
; RUN: opt 1.ll -o 1.bc
; RUN: opt 2.ll -o d/2.bc

; RUN: rm -f objpath.o
; RUN: ld.lld --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM
; RUN: llvm-objdump -d objpath.o | FileCheck %s
; RUN: ls 3* objpath* | count 2

; RUN: rm -f 3 objpath.o
; RUN: ld.lld --thinlto-index-only=3.txt --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o 3
; RUN: llvm-objdump -d objpath.o | FileCheck %s
; RUN: not ls 3

; NM: T f
; NM: T g

; CHECK: file format elf64-x86-64
; CHECK: <f>:
; CHECK: <g>:

;; Test --lto-obj-path= for ThinLTO.
; RUN: opt -module-summary 1.ll -o 1.bc
; RUN: opt -module-summary 2.ll -o d/2.bc

; RUN: ld.lld --plugin-opt=obj-path=objpath.o -shared 1.bc d/2.bc -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM3
; RUN: llvm-objdump -d objpath.o1 | FileCheck %s --check-prefix=CHECK1
; RUN: llvm-objdump -d objpath.o2 | FileCheck %s --check-prefix=CHECK2

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
; RUN: rm -f objpath.o objpath.o1 objpath.o2
; RUN: ld.lld --thinlto-index-only --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o /dev/null
; RUN: llvm-objdump -d objpath.o | FileCheck %s --check-prefix=EMPTY
; RUN: not ls objpath.o1
; RUN: not ls objpath.o2

;; Test --plugin-opt=obj-path=.
; RUN: rm -f objpath.o
; RUN: ld.lld --plugin-opt=thinlto-index-only --plugin-opt=obj-path=objpath.o -shared 1.bc d/2.bc -o /dev/null
; RUN: llvm-objdump -d objpath.o | FileCheck %s --check-prefix=EMPTY

;; Ensure lld emits empty combined module if specific obj-path.
; RUN: mkdir obj
; RUN: ld.lld --plugin-opt=obj-path=objpath.o -shared 1.bc d/2.bc -o obj/out --save-temps
; RUN: ls obj/out.lto.o obj/out1.lto.o obj/out2.lto.o

;; Ensure lld does not emit empty combined module by default.
; RUN: rm -fr obj && mkdir obj
; RUN: ld.lld -shared 1.bc d/2.bc -o obj/out --save-temps
; RUN: ls obj/out*.lto.* | count 2

; EMPTY:     file format elf64-x86-64
; EMPTY-NOT: {{.}}

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- 2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @g() {
entry:
  ret void
}
