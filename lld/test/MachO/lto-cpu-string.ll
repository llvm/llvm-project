; REQUIRES: x86
; RUN: llvm-as %s -o %t.o

; RUN: %lld %t.o -o %t.dylib -dylib
; RUN: llvm-objdump -d --section="__text" --no-leading-addr --no-show-raw-insn %t.dylib | FileCheck %s
; CHECK: nop{{$}}

; RUN: %lld -mcpu znver1 %t.o -o %t.znver1.dylib -dylib
; RUN: llvm-objdump -d --section="__text" --no-leading-addr --no-show-raw-insn %t.znver1.dylib | FileCheck %s --check-prefix=ZNVER1

; ZNVER1: nopw
; ZNVER1-NOT: nop{{$}}

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() #0 {
entry:
  call void asm sideeffect ".p2align        4", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

attributes #0 = { "frame-pointer"="all" }
