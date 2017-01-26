; RUN: llvm-as -o %t.obj %s
; RUN: lld-link /dll /out:%t.dll %t.obj
; RUN: llvm-objdump -d %t.dll | FileCheck %s

; Checks that code for foo is emitted, as required by the /INCLUDE directive.
; CHECK: xorl %eax, %eax
; CHECK-NEXT: retq

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @_DllMainCRTStartup() {
  ret void
}

define i32 @foo() {
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 6, !"Linker Options", !1}
!1 = !{!2}
!2 = !{!"/INCLUDE:foo"}
