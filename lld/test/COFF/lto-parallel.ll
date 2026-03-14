; REQUIRES: x86
; RUN: llvm-as -o %t.obj %s
; RUN: lld-link -opt:noicf /out:%t.exe /entry:foo /include:bar /opt:lldltopartitions=2 /subsystem:console /lldmap:%t.map %t.obj /debug /pdb:%t.pdb
; RUN: FileCheck %s < %t.map
; RUN: llvm-pdbutil dump %t.pdb --modules | FileCheck %s --check-prefix=PDB

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK: lto-parallel.ll.tmp.exe.lto.obj:
; CHECK: lto-parallel.ll.tmp.exe.lto.obj:
; CHECK-NEXT: foo
define void @foo() {
  call void @bar()
  ret void
}

; CHECK: lto-parallel.ll.tmp.exe.lto.1.obj:
; CHECK: lto-parallel.ll.tmp.exe.lto.1.obj:
; CHECK: bar
define void @bar() {
  call void @foo()
  ret void
}


; Objects in the PDB should receive distinct names.

; PDB: Modules
; PDB: Mod 0000 | `{{.*}}lto-parallel.ll.tmp.exe.lto.obj`:
; PDB: Obj: `{{.*}}lto-parallel.ll.tmp.exe.lto.obj`:
; PDB: Mod 0001 | `{{.*}}lto-parallel.ll.tmp.exe.lto.1.obj`:
; PDB: Obj: `{{.*}}lto-parallel.ll.tmp.exe.lto.1.obj`:
; PDB: Mod 0002 | `* Linker *`:
; PDB: Obj: ``:
