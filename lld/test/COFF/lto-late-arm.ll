; REQUIRES: arm

;; A bitcode file can generate undefined references to symbols that weren't
;; listed as undefined on the bitcode file itself, when lowering produces
;; calls to e.g. builtin helper functions. If these functions are provided
;; as LTO bitcode, the linker would hit an unhandled state. (In practice,
;; compiler-rt builtins are always compiled with -fno-lto, so this shouldn't
;; happen.)

; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: llvm-as %t.dir/main.ll -o %t.main.obj
; RUN: llvm-as %t.dir/sdiv.ll -o %t.sdiv.obj
; RUN: llvm-ar rcs %t.sdiv.lib %t.sdiv.obj

; RUN: env LLD_IN_TEST=1 not lld-link /entry:entry %t.main.obj %t.sdiv.lib /out:%t.exe /subsystem:console 2>&1 | FileCheck %s

; CHECK: error: LTO object file lto-late-arm.ll.tmp.sdiv.lib(lto-late-arm.ll.tmp.sdiv.obj) linked in after doing LTO compilation.

;--- main.ll
target datalayout = "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-w64-windows-gnu"

@num = dso_local global i32 100

define dso_local arm_aapcs_vfpcc i32 @entry(i32 %param) {
entry:
  %0 = load i32, ptr @num
  %div = sdiv i32 %0, %param
  ret i32 %div
}
;--- sdiv.ll
target datalayout = "e-m:w-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7-w64-windows-gnu"

define dso_local arm_aapcs_vfpcc void @__rt_sdiv() {
entry:
  ret void
}
