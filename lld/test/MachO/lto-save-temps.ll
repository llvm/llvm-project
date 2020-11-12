; REQUIRES: x86

; Test that we compile regular LTO inputs in a single task but handle ThinLTO
; modules in separate tasks.

; RUN: rm -rf %t; split-file %s %t
; RUN: llvm-as %t/foo.ll -o %t/foo.o
; RUN: llvm-as %t/test.ll -o %t/test.o
; RUN: %lld -save-temps %t/foo.o %t/test.o -o %t/test
; RUN: llvm-objdump -d --no-show-raw-insn %t/test.lto.o | FileCheck %s --check-prefix=ALL
; RUN: llvm-objdump -d --no-show-raw-insn %t/test | FileCheck %s --check-prefix=ALL

; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-summary %t/foo.ll -o %t/foo.o
; RUN: opt -module-summary %t/test.ll -o %t/test.o
; RUN: %lld -save-temps %t/foo.o %t/test.o -o %t/test
; RUN: llvm-objdump -d --no-show-raw-insn %t/test1.lto.o | FileCheck %s --check-prefix=FOO
; RUN: llvm-objdump -d --no-show-raw-insn %t/test2.lto.o | FileCheck %s --check-prefix=MAIN
; RUN: llvm-objdump -d --no-show-raw-insn %t/test | FileCheck %s --check-prefix=ALL

; FOO:      <_foo>:
; FOO-NEXT: retq

; MAIN:      <_main>:
; MAIN-NEXT: retq

; ALL:      <_foo>:
; ALL-NEXT: retq
; ALL:      <_main>:
; ALL-NEXT: retq

;--- foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

;--- test.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}
