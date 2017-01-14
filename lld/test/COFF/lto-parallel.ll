; RUN: llvm-as -o %t.obj %s
; RUN: lld-link /out:%t.exe /entry:foo /include:bar /opt:lldltojobs=2 /subsystem:console /lldmap:%t.map %t.obj
; RUN: FileCheck %s < %t.map

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK: <lto object>
; CHECK-NEXT: foo
define void @foo() {
  call void @bar()
  ret void
}

; CHECK: <lto object>
; CHECK-NEXT: bar
define void @bar() {
  call void @foo()
  ret void
}
