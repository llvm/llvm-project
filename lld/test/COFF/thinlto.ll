; REQUIRES: x86
; RUN: rm -fr %T/thinlto
; RUN: mkdir %T/thinlto
; RUN: opt -thinlto-bc -o %T/thinlto/main.obj %s
; RUN: opt -thinlto-bc -o %T/thinlto/foo.obj %S/Inputs/lto-dep.ll
; RUN: lld-link /lldsavetemps /out:%T/thinlto/main.exe /entry:main /subsystem:console %T/thinlto/main.obj %T/thinlto/foo.obj
; RUN: llvm-nm %T/thinlto/main.exe.lto.foo.obj | FileCheck %s

; Test various possible options for /opt:lldltojobs
; RUN: lld-link /lldsavetemps /out:%T/thinlto/main.exe /entry:main /subsystem:console %T/thinlto/main.obj %T/thinlto/foo.obj /opt:lldltojobs=1
; RUN: llvm-nm %T/thinlto/main.exe.lto.foo.obj | FileCheck %s
; RUN: lld-link /lldsavetemps /out:%T/thinlto/main.exe /entry:main /subsystem:console %T/thinlto/main.obj %T/thinlto/foo.obj /opt:lldltojobs=all
; RUN: llvm-nm %T/thinlto/main.exe.lto.foo.obj | FileCheck %s
; RUN: lld-link /lldsavetemps /out:%T/thinlto/main.exe /entry:main /subsystem:console %T/thinlto/main.obj %T/thinlto/foo.obj /opt:lldltojobs=100
; RUN: llvm-nm %T/thinlto/main.exe.lto.foo.obj | FileCheck %s
; RUN: not lld-link /lldsavetemps /out:%T/thinlto/main.exe /entry:main /subsystem:console %T/thinlto/main.obj %T/thinlto/foo.obj /opt:lldltojobs=foo 2>&1 | FileCheck %s --check-prefix=BAD-JOBS
; BAD-JOBS: error: /opt:lldltojobs: invalid job count: foo

; This command will store full path to foo.obj in the archive %t.lib
; Check that /lldsavetemps is still usable in such case.
; RUN: lld-link /lib %T/thinlto/foo.obj /out:%t.lib
; RUN: lld-link /lldsavetemps /out:%t.exe /entry:main /subsystem:console %T/thinlto/main.obj %t.lib

; CHECK-NOT: U foo

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @main() {
  call void @foo()
  ret i32 0
}

declare void @foo()
