;; Copied from testr/ELF/lto/obj-path.ll
;; Test --lto-obj-path= for regular LTO.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: mkdir d
; RUN: opt 1.ll -o 1.bc
; RUN: opt 2.ll -o d/2.bc

; RUN: rm -f objpath.o
; RUN: wasm-ld --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM
; RUN: llvm-objdump -d objpath.o | FileCheck %s
; RUN: ls 3* objpath* | count 2

; RUN: rm -f 3 objpath.o
; RUN: wasm-ld --thinlto-index-only=3.txt --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o 3
; RUN: llvm-objdump -d objpath.o | FileCheck %s
; RUN: not ls 3

; NM: T f
; NM: T g

; CHECK: file format wasm
; CHECK: <f>:
; CHECK: <g>:

;; Test --lto-obj-path= for ThinLTO.
; RUN: opt -module-summary 1.ll -o 1.bc
; RUN: opt -module-summary 2.ll -o d/2.bc

; RUN: wasm-ld --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o 3
; RUN: llvm-nm 3 | FileCheck %s --check-prefix=NM3
; RUN: llvm-objdump -d objpath.o1 | FileCheck %s --check-prefix=CHECK1
; RUN: llvm-objdump -d objpath.o2 | FileCheck %s --check-prefix=CHECK2

; NM3:      T f
; NM3-NEXT: T g

; CHECK1:       file format wasm
; CHECK1-EMPTY:
; CHECK1-NEXT:  Disassembly of section CODE:
; CHECK1:  <f>:
; CHECK1-EMPTY:
; CHECK1-NEXT:    end
; CHECK1-NOT:   {{.}}

; CHECK2:       file format wasm
; CHECK2-EMPTY:
; CHECK2-NEXT:  Disassembly of section CODE:
; CHECK2:  <g>:
; CHECK2-EMPTY:
; CHECK2-NEXT:    end
; CHECK2-NOT:   {{.}}

;; With --thinlto-index-only, --lto-obj-path= creates just one file.
; RUN: rm -f objpath.o objpath.o1 objpath.o2
; RUN: wasm-ld --thinlto-index-only --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o /dev/null
; RUN: llvm-objdump -d objpath.o | FileCheck %s --check-prefix=EMPTY
; RUN: not ls objpath.o1
; RUN: not ls objpath.o2

;; Ensure lld emits empty combined module if specific obj-path.
; RUN: mkdir obj
; RUN: wasm-ld --lto-obj-path=objpath.o -shared 1.bc d/2.bc -o obj/out --save-temps
; RUN: ls obj/out.lto.o out.lto.1.o d/out.lto.2.o

;; Ensure lld does not emit empty combined module by default.
; RUN: rm -fr obj && mkdir obj
; RUN: wasm-ld -shared 1.bc d/2.bc -o obj/out --save-temps
; RUN: not test -e obj/out.lto.o

; EMPTY:     file format wasm
; EMPTY-NOT: {{.}}

;--- 1.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- 2.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define void @g() {
entry:
  ret void
}
