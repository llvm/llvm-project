; REQUIRES: x86
; RUN: rm -fr %t.dir/savetemps
; RUN: mkdir -p %t.dir/savetemps
; RUN: llvm-as -o %t.dir/savetemps/savetemps.obj %s
; RUN: lld-link /out:%t.dir/savetemps/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps/savetemps.obj
; RUN: not llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.0.preopt.bc
; RUN: not llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.2.internalize.bc
; RUN: not llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.4.opt.bc
; RUN: not llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.5.precodegen.bc
; RUN: not llvm-objdump -s %t.dir/savetemps/savetemps.exe.lto.obj
; RUN: lld-link /lldsavetemps /out:%t.dir/savetemps/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps/savetemps.obj
; RUN: llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.0.preopt.bc | FileCheck %s
; RUN: llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.2.internalize.bc | FileCheck %s
; RUN: llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.4.opt.bc | FileCheck %s
; RUN: llvm-dis -o - %t.dir/savetemps/savetemps.exe.0.5.precodegen.bc | FileCheck %s
; RUN: llvm-objdump -s %t.dir/savetemps/savetemps.exe.lto.obj | \
; RUN:     FileCheck --check-prefix=CHECK-OBJDUMP %s

; CHECK: define {{(noundef )?}}i32 @main()
; CHECK-OBJDUMP: file format coff

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @main() {
  ret i32 0
}
