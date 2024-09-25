; REQUIRES: x86
; RUN: llvm-as %s -o %t.obj

; RUN: lld-link %t.obj -noentry -nodefaultlib -out:%t.dll -dll
; RUN: llvm-objdump -d --section=".text" --no-leading-addr --no-show-raw-insn %t.dll | FileCheck %s
; CHECK: nop{{$}}

; RUN: lld-link -mllvm:-mcpu=znver1 -noentry -nodefaultlib %t.obj -out:%t.znver1.dll -dll
; RUN: llvm-objdump -d --section=".text" --no-leading-addr --no-show-raw-insn %t.znver1.dll | FileCheck --check-prefix=ZNVER1 %s
; ZNVER1: nopw

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26433"

define dllexport void @foo() #0 {
entry:
  call void asm sideeffect ".p2align        4", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

attributes #0 = { "frame-pointer"="all" }
