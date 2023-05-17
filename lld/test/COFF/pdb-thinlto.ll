; REQUIRES: x86
; RUN: rm -rf %t && mkdir -p %t && cd %t
; RUN: opt -thinlto-bc -o main.bc %s
; RUN: opt -thinlto-bc -o foo.bc %S/Inputs/lto-dep.ll

; Even if the native object is cached, the PDB must be the same.
; RUN: rm -rf thinltocachedir && mkdir thinltocachedir

; RUN: lld-link /lldltocache:thinltocachedir /out:main.exe /entry:main /subsystem:console main.bc foo.bc /debug /pdb:main.pdb

; RUN: llvm-pdbutil dump --modules main.pdb | FileCheck %s

; Run again with the cache. Make sure we get the same object names.

; RUN: lld-link /lldltocache:thinltocachedir /out:main.exe /entry:main /subsystem:console main.bc foo.bc /debug /pdb:main.pdb

; RUN: llvm-pdbutil dump --modules main.pdb | FileCheck %s


target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @main() {
  call void @foo()
  ret i32 0
}

declare void @foo()

; CHECK:                           Modules
; CHECK: ============================================================
; CHECK: Mod 0000 | `{{.*}}main.exe.lto.main.bc`:
; CHECK: Obj: `{{.*}}main.exe.lto.main.bc`:
; CHECK: Mod 0001 | `{{.*}}main.exe.lto.foo.bc`:
; CHECK: Obj: `{{.*}}main.exe.lto.foo.bc`:
; CHECK: Mod 0002 | `* Linker *`:
