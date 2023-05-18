; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t -mllvm -mcpu=znver1 -mllvm -debug-pass=Structure -mllvm -print-after-all 2>&1 | FileCheck %s
; RUN: llvm-objdump -d -j .text %t | FileCheck %s --check-prefix=DISASM

;; We support -plugin-opt=- for LLVMgold.so compatibility. With a few exceptions,
;; most -plugin-opt=- prefixed options are passed through to cl::ParseCommandLineOptions.
; RUN: ld.lld %t.o -o %t -plugin-opt=-debug-pass=Structure -plugin-opt=-print-after-all 2>&1 | FileCheck %s

; CHECK: Pass Arguments:
; CHECK: # *** IR Dump

; DISASM: nopw

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() #0 {
entry:
  call void asm sideeffect ".p2align 4, 0x90", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

attributes #0 = { "frame-pointer"="all" }
