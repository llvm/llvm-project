; REQUIRES: x86
; RUN: split-file %s %t.dir
;; LTO
; RUN: llvm-as %t.dir/main.ll -o %t.main.bc
; RUN: llvm-as %t.dir/wrap.ll -o %t.wrap.bc
; RUN: llvm-as %t.dir/other.ll -o %t.other.bc
; RUN: rm -f %t.bc.lib
; RUN: llvm-ar rcs %t.bc.lib %t.wrap.bc %t.other.bc
;; ThinLTO
; RUN: opt -module-summary %t.dir/main.ll -o %t.main.thin
; RUN: opt -module-summary %t.dir/wrap.ll -o %t.wrap.thin
; RUN: opt -module-summary %t.dir/other.ll -o %t.other.thin
; RUN: rm -f %t.thin.lib
; RUN: llvm-ar rcs %t.thin.lib %t.wrap.thin %t.other.thin
;; Object
; RUN: llc %t.dir/main.ll -o %t.main.obj --filetype=obj
; RUN: llc %t.dir/wrap.ll -o %t.wrap.obj --filetype=obj
; RUN: llc %t.dir/other.ll -o %t.other.obj --filetype=obj
; RUN: rm -f %t.obj.lib
; RUN: llvm-ar rcs %t.obj.lib %t.wrap.obj %t.other.obj

;; This test verifies that -wrap works correctly for inter-module references to
;; the wrapped symbol, when LTO or ThinLTO is involved. It checks for various
;; combinations of bitcode and regular objects.

;; LTO + LTO
; RUN: lld-link -out:%t.bc-bc.exe %t.main.bc -libpath:%T %t.bc.lib -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: llvm-objdump -d %t.bc-bc.exe | FileCheck %s --check-prefixes=CHECK,JMP

;; LTO + Object
; RUN: lld-link -out:%t.bc-obj.exe %t.main.bc -libpath:%T %t.obj.lib -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: llvm-objdump -d %t.bc-obj.exe | FileCheck %s --check-prefixes=CHECK,JMP

;; Object + LTO
; RUN: lld-link -out:%t.obj-bc.exe %t.main.obj -libpath:%T %t.bc.lib -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: llvm-objdump -d %t.obj-bc.exe | FileCheck %s --check-prefixes=CHECK,CALL

;; ThinLTO + ThinLTO
; RUN: lld-link -out:%t.thin-thin.exe %t.main.thin -libpath:%T %t.thin.lib -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: llvm-objdump -d %t.thin-thin.exe | FileCheck %s --check-prefixes=CHECK,JMP

;; ThinLTO + Object
; RUN: lld-link -out:%t.thin-obj.exe %t.main.thin -libpath:%T %t.obj.lib -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: llvm-objdump -d %t.thin-obj.exe | FileCheck %s --check-prefixes=CHECK,JMP

;; Object + ThinLTO
; RUN: lld-link -out:%t.obj-thin.exe %t.main.obj -libpath:%T %t.thin.lib -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: llvm-objdump -d %t.obj-thin.exe | FileCheck %s --check-prefixes=CHECK,CALL

;; Make sure that calls in entry() are not eliminated and that bar is
;; routed to __wrap_bar.

; CHECK: <entry>:
; CHECK: {{jmp|callq}}{{.*}}<__wrap_bar>

;--- main.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

declare void @bar()

define void @entry() {
  call void @bar()
  ret void
}

;--- wrap.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

declare void @other()

define void @__wrap_bar() {
  call void @other()
  ret void
}

;--- other.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define void @other() {
  ret void
}
