; REQUIRES: x86

;; Verify that we successfully merge weak definitions across bitcode and regular
;; assembly files, even when EH frames are present. We would previously
;; segfault.

; RUN: rm -rf %t; split-file %s %t
; RUN: llvm-as %t/foo-1.ll -o %t/foo-1.o

;; When changing the assembly input, uncomment these lines to re-generate the
;; YAML.
; COM: llvm-mc --emit-dwarf-unwind=always -filetype=obj -triple=x86_64-apple-darwin %t/foo-2.s -o %t/foo-2.o
; COM: ld -r %t/foo-2.o -o %t/foo-2-r.o
; COM: obj2yaml %t/foo-2-r.o -o %S/Inputs/lto-obj-weak-def.yaml

; RUN: yaml2obj %S/Inputs/lto-obj-weak-def.yaml -o %t/foo-2-r.o 
; RUN: %lld -lSystem -dylib %t/foo-1.o %t/foo-2-r.o -o /dev/null

;--- foo-1.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define weak void @foo() {
  ret void
}

;--- foo-2.s
.globl _foo
.weak_definition _foo
_foo:
  .cfi_startproc
  .cfi_def_cfa_offset 8
  ret
  .cfi_endproc
