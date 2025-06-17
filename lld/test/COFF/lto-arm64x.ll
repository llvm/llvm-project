; REQUIRES: aarch64, x86
; RUN: split-file %s %t.dir && cd %t.dir

; RUN: llvm-as arm64ec.ll -o arm64ec.obj
; RUN: llvm-as aarch64.ll -o aarch64.obj
; RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-arm64.obj
; RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj

; RUN: lld-link -machine:arm64x aarch64.obj arm64ec.obj loadconfig-arm64.obj loadconfig-arm64ec.obj -out:out.exe -subsystem:console
; RUN: llvm-objdump -d out.exe | FileCheck %s

; CHECK:      0000000140001000 <.text>:
; CHECK-NEXT: 140001000: 52800020     mov     w0, #0x1                // =1
; CHECK-NEXT: 140001004: d65f03c0     ret
; CHECK-NEXT:                 ...
; CHECK-NEXT: 140002000: 00000009     udf     #0x9
; CHECK-NEXT: 140002004: 52800040     mov     w0, #0x2                // =2
; CHECK-NEXT: 140002008: d65f03c0     ret

; CHECK:      0000000140003000 <.hexpthk>:
; CHECK-NEXT: 140003000: 48 8b c4                     movq    %rsp, %rax
; CHECK-NEXT: 140003003: 48 89 58 20                  movq    %rbx, 0x20(%rax)
; CHECK-NEXT: 140003007: 55                           pushq   %rbp
; CHECK-NEXT: 140003008: 5d                           popq    %rbp
; CHECK-NEXT: 140003009: e9 f6 ef ff ff               jmp     0x140002004 <.text+0x1004>
; CHECK-NEXT: 14000300e: cc                           int3
; CHECK-NEXT: 14000300f: cc                           int3

#--- arm64ec.ll

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64ec-unknown-windows-msvc"

define dso_local i32 @mainCRTStartup() {
entry:
  ret i32 2
}

#--- aarch64.ll

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-windows-msvc"

define dso_local i32 @mainCRTStartup() {
entry:
  ret i32 1
}
