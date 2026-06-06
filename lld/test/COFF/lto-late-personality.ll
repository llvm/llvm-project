; REQUIRES: x86

;; A bitcode file can generate undefined references to symbols that weren't
;; listed as undefined on the bitcode file itself, if there's a reference to
;; an unexpected personality routine via asm(). If the personality function
;; is provided as LTO bitcode, the linker would hit an unhandled state.

; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: llvm-as %t.dir/main.ll -o %t.main.obj
; RUN: llvm-as %t.dir/other.ll -o %t.other.obj
; RUN: llvm-as %t.dir/personality.ll -o %t.personality.obj
; RUN: llvm-ar rcs %t.personality.lib %t.personality.obj

; RUN: env LLD_IN_TEST=1 not lld-link /entry:entry %t.main.obj %t.other.obj %t.personality.lib /out:%t.exe /subsystem:console /opt:lldlto=0 /debug:symtab 2>&1 | FileCheck %s

; CHECK: error: LTO object file lto-late-personality.ll.tmp.personality.lib(lto-late-personality.ll.tmp.personality.obj) linked in after doing LTO compilation.

;--- main.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define i32 @entry() {
entry:
  tail call void @other()
  tail call void asm sideeffect ".seh_handler __C_specific_handler, @except\0A", "~{dirflag},~{fpsr},~{flags}"()
  ret i32 0
}

declare dso_local void @other()

;--- other.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define dso_local void @other() {
entry:
  ret void
}

;--- personality.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define void @__C_specific_handler() {
entry:
  ret void
}
