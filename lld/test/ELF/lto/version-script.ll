; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: echo "VERSION_1.0{ global: foo; local: *; }; VERSION_2.0{ global: bar; local: *; };" > %t.script
; RUN: ld.lld %t.o -o %t2 -shared --version-script %t.script -save-temps
; RUN: llvm-dis < %t2.0.0.preopt.bc | FileCheck %s
; RUN: llvm-readelf --dyn-syms %t2 | FileCheck --check-prefix=DSO %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

; CHECK: define void @foo()
; CHECK: define void @bar()

; DSO: Symbol table '.dynsym' contains 3 entries:
; DSO:      1: {{.*}}               1 FUNC    GLOBAL DEFAULT [[#]] foo@@VERSION_1.0{{$}}
; DSO:      2: {{.*}}               1 FUNC    GLOBAL DEFAULT [[#]] bar@@VERSION_2.0{{$}}
