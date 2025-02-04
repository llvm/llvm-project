; REQUIRES: aarch64, x86

; RUN: llvm-as %s -o %t.obj
; RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadconfig.obj

; RUN: lld-link -machine:arm64ec %t.obj %t-loadconfig.obj -out:%t.exe -subsystem:console
; RUN: llvm-objdump -d %t.exe | FileCheck %s

; CHECK:      0000000140001000 <.text>:
; CHECK-NEXT: 140001000: 00000009     udf     #0x9
; CHECK-NEXT: 140001004: 52800020     mov     w0, #0x1                // =1
; CHECK-NEXT: 140001008: d65f03c0     ret

; CHECK:      0000000140002000 <.hexpthk>:
; CHECK-NEXT: 140002000: 48 8b c4                     movq    %rsp, %rax
; CHECK-NEXT: 140002003: 48 89 58 20                  movq    %rbx, 0x20(%rax)
; CHECK-NEXT: 140002007: 55                           pushq   %rbp
; CHECK-NEXT: 140002008: 5d                           popq    %rbp
; CHECK-NEXT: 140002009: e9 f6 ef ff ff               jmp     0x140001004 <.text+0x4>
; CHECK-NEXT: 14000200e: cc                           int3
; CHECK-NEXT: 14000200f: cc                           int3

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64ec-unknown-windows-msvc"

define dso_local i32 @mainCRTStartup() {
entry:
  ret i32 1
}
