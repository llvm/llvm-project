; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/swift-foo.s -o %t/swift-foo.o
; RUN: llvm-ar rcs %t/libswiftFoo.a %t/swift-foo.o
; RUN: llvm-as %t/lc-linker-opt.ll -o %t/lc-linker-opt.o
; RUN: llvm-as %t/no-lc-linker-opt.ll -o %t/no-lc-linker-opt.o

; RUN: %lld -lSystem -force_load_swift_libs -L%t %t/lc-linker-opt.o -o %t/lc-linker-opt
; RUN: llvm-objdump --macho --syms %t/lc-linker-opt | FileCheck %s --check-prefix=HAS-SWIFT

; RUN: %lld -lSystem -L%t %t/lc-linker-opt.o -o %t/lc-linker-opt-no-force
; RUN: llvm-objdump --macho --syms %t/lc-linker-opt-no-force | FileCheck %s --check-prefix=NO-SWIFT

;; Swift libraries passed on the CLI don't get force-loaded!
; RUN: %lld -lSystem -force_load_swift_libs -lswiftFoo -L%t %t/no-lc-linker-opt.o -o %t/no-lc-linker-opt
; RUN: llvm-objdump --macho --syms %t/no-lc-linker-opt | FileCheck %s --check-prefix=NO-SWIFT

; HAS-SWIFT: _swift_foo
; NO-SWIFT-NOT: _swift_foo

;--- lc-linker-opt.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-lswiftFoo"}
!llvm.linker.options = !{!0}

define void @main() {
  ret void
}

;--- no-lc-linker-opt.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}

;--- swift-foo.s
.globl _swift_foo
_swift_foo:
