; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: opt --thinlto-bc %t/f.ll -o %t/f.o
; RUN: llvm-ar rcs %t/libf.a %t/f.o
; RUN: opt --thinlto-bc %t/q.ll -o %t/q.o
; RUN: llvm-ar rcs %t/libq.a %t/q.o
; RUN: opt --thinlto-bc %t/b.ll -o %t/b.o
; RUN: llvm-ar rcs %t/libb.a %t/b.o
; RUN: opt --thinlto-bc %t/m.ll -o %t/m.o

; RUN: not %no-fatal-warnings-lld -lSystem %t/libf.a %t/libq.a %t/libb.a %t/m.o -o %t/test.out 2>&1 | FileCheck %s

; We can't read symbols that are set by `module asm` in the bitcode's symbol table.
; LTO compiles only f.ll, b.ll, and m.ll without q.ll due to missing reference to `q_asm` in f.ll.
; However, later `q_asm` appears in the resulting object for `f.ll`, which triggers resolving
; a ODR symbol `q_odr` from `q.ll,` as opposed to one expected from `b.ll`.
; As a result, we have invalid relocation for `q_odr` as `q.ll` is neither compiled, nor linked.

; CHECK: error: The pending prevailing symbol(_q_odr) in the bitcode file({{.*}}libb.a(b.o)) is overridden by a non-native object (from bitcode): {{.*}}libq.a(q.o)

;--- f.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

module asm ".no_dead_strip _q_asm"

define i64 @f() {
  ret i64 1
}

;--- q.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@q_asm = global i64 1

@q_odr = linkonce_odr hidden unnamed_addr constant i64 0


;--- b.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@q_odr = linkonce_odr hidden unnamed_addr constant i64 1

define i64 @b() {
  ret i64 ptrtoint (i64* @q_odr to i64)
}

;--- m.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

declare i64 @f()
declare i64 @b()

define i64 @main() {
  %call1 = call i64 @f()
  %call2 = call i64 @b()
  %add = add nsw i64 %call1, %call2
  ret i64 %add
}
