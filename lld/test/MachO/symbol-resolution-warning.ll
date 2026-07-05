; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: opt %t/f.ll -o %t/f.o
; RUN: llvm-ar rcs %t/libf.a %t/f.o
; RUN: llc -filetype=obj %t/q.ll -o %t/q.o
; RUN: llvm-ar rcs %t/libq.a %t/q.o
; RUN: llc -filetype=obj %t/m.ll -o %t/m.o

; RUN: %no-fatal-warnings-lld -dylib -lSystem -L%t %t/libf.a %t/m.o -o %t/test.out 2>&1 | FileCheck %s

; We can't read symbols that are set by module asm in the bitcode's symbol table.
; LTO internalizes `odr` in `f.ll`, and thus the prevailing `odr` remains unresolved.
; `q.ll` is loaded after LTO is finished during post-processing LC linker options.
; The unresolved prevailing `odr` is now resolved with `q.ll`.

; CHECK: warning: The pending prevailing symbol(_odr) in the bitcode file({{.*}}libf.a(f.o)) is overridden by a post-processed native object (from native archive): {{.*}}libq.a(q.o)

;--- f.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

!0 = !{!"-lq"}
!llvm.linker.options = !{!0}

define i64 @f()  {
  %1 = call i64 @odr()
  %2 = call i64 @q()
  %3 = add i64 %1, %2
  ret i64 %3
}

define weak hidden i64 @odr() noinline {
  ret i64 1
}

declare i64 @q()

;--- q.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define i64 @q() {
  %1 = call i64 @odr()
  ret i64 %1
}

define linkonce_odr hidden i64 @odr() noinline {
  ret i64 2
}

;--- m.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

declare i64 @f()

define i64 @m() {
  %1 = call i64 @f()
  ret i64 %1
}
