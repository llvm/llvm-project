; REQUIRES: x86
; RUN: llvm-as -o %main.obj %s
; RUN: lld-link /out:%main.exe /entry:main  /subsystem:console %main.obj
; RUN: llvm-readobj --coff-debug-directory %main.exe
; CHECK: {{.*}}GCTL{{.*}}

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "aarch64-pc-windows-msvc"

define i32 @main() {
  ret i32 0
}
