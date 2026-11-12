; REQUIRES: arm

; RUN: llvm-as %s -o %t.obj

; RUN: lld-link /entry:entry %t.obj /out:%t.exe /subsystem:console 2>&1 | FileCheck %s --check-prefix=ERR --allow-empty
; RUN: llvm-readobj %t.exe | FileCheck %s

; ERR-NOT: /machine is not specified

; CHECK: Format: COFF-ARM{{$}}
; CHECK: Arch: thumb

target datalayout = "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-w64-windows-gnu"

define dso_local arm_aapcs_vfpcc void @entry() {
entry:
  ret void
}
