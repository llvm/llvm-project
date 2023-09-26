; REQUIRES: aarch64
; RUN: rm -rf %t; split-file %s %t

;; Test that a weak symbol in a direct .o file wins over
;; a weak symbol in a .a file.
;; Like weak-definition-in-main-file.s, but in bitcode.

; RUN: llvm-as %t/test.ll -o %t/test.o
; RUN: llvm-as %t/weakfoo.ll -o %t/weakfoo.o

; RUN: llvm-ar --format=darwin rcs %t/weakfoo.a %t/weakfoo.o

; PREFER-DIRECT-OBJECT-NOT: O __TEXT,weak _foo

; RUN: %lld -lSystem -o %t/out %t/weakfoo.a %t/test.o
; RUN: llvm-objdump --syms %t/out | FileCheck %s --check-prefix=PREFER-DIRECT-OBJECT

;--- weakfoo.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @baz() noinline optnone {
  ret void
}

define weak void @foo() noinline optnone section "__TEXT,weak" {
  ret void
}

;--- test.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @baz();

define weak void @foo() noinline optnone {
  ret void
}

define void @main() {
  ; This pulls in weakfoo.a due to the __baz undef, but __foo should
  ; still be resolved against the weak symbol in this file.
  call void @baz()
  call void @foo()
  ret void
}
