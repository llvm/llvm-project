; REQUIRES: x86
; RUN: llvm-as -o %t-weak.obj %S/Inputs/weak-def.ll
; RUN: llvm-as -o %t-strong.obj %S/Inputs/strong-def.ll
; RUN: llvm-as -o %t.obj %s
; RUN: lld-link /dll /out:%t-weak-first.dll %t.obj %t-weak.obj %t-strong.obj
; RUN: lld-link /dll /out:%t-strong-first.dll %t.obj %t-strong.obj %t-weak.obj
; RUN: lld-link /dll /out:%t-weak-only.dll %t.obj %t-weak.obj
; RUN: llvm-objdump --no-print-imm-hex -d %t-weak-first.dll | FileCheck --check-prefix=CHECK-STRONG %s
; RUN: llvm-objdump --no-print-imm-hex -d %t-strong-first.dll | FileCheck --check-prefix=CHECK-STRONG %s
; RUN: llvm-objdump --no-print-imm-hex -d %t-weak-only.dll | FileCheck --check-prefix=CHECK-WEAK %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare noundef i32 @foo() local_unnamed_addr

define dllexport i32 @bar() local_unnamed_addr {
  %1 = tail call noundef i32 @foo()
  ret i32 %1
}

define void @_DllMainCRTStartup() {
entry:
  ret void
}

; CHECK-STRONG: movl $5678, %eax
; CHECK-STRONG-NOT: movl $1234, %eax
; CHECK-WEAK: movl $1234, %eax
; CHECK-WEAK-NOT: movl $5678, %eax
