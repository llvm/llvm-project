; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t

; LTO
; RUN: llvm-as %s -o a.o
; RUN: llvm-as %S/Inputs/defsym-bar.ll -o b.o
; RUN: ld.lld a.o b.o -shared -o a.so -defsym=bar2=bar3 -save-temps
; RUN: llvm-readelf --symbols a.so.lto.o | FileCheck --check-prefix=OBJ %s
; RUN: llvm-objdump -d a.so | FileCheck %s

; ThinLTO
; RUN: opt -module-summary %s -o a.o
; RUN: opt -module-summary %S/Inputs/defsym-bar.ll -o b.o
; RUN: ld.lld a.o b.o -shared -o a2.so -defsym=bar2=bar3 -save-temps
; RUN: llvm-readelf --symbols a2.so.lto.a.o | FileCheck --check-prefix=OBJ %s
; RUN: llvm-objdump -d a2.so | FileCheck %s

; OBJ:  UND bar2

; Call to bar2() should not be inlined and should be routed to bar3()
; Symbol bar3 should not be eliminated

; CHECK:      <foo>:
; CHECK-NEXT: pushq %rax
; CHECK-NEXT: callq
; CHECK-NEXT: callq{{.*}}<bar3>
; CHECK-NEXT: popq %rax
; CHECK-NEXT: jmp

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar1()
declare void @bar2()
declare void @bar3()

define void @foo() {
  call void @bar1()
  call void @bar2()
  call void @bar3()
  ret void
}
