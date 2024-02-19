; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: llc -filetype=obj %t/q.ll -o %t/q.o
; RUN: llvm-ar cru %t/libq.a %t/q.o

; RUN: llc -filetype=obj %t/f.ll -o %t/f.nolto.o
; RUN: opt --thinlto-bc %t/f.ll -o %t/f.thinlto.o
; RUN: opt %t/f.ll -o %t/f.lto.o

; RUN: llc -filetype=obj %t/b.ll -o %t/b.nolto.o
; RUN: opt --thinlto-bc %t/b.ll -o %t/b.thinlto.o
; RUN: opt %t/b.ll -o %t/b.lto.o

; (1) NoLTO-NoLTO
; RUN: %lld -dylib -lSystem -L%t %t/f.nolto.o %t/b.nolto.o -o %t/nolto-nolto.out
; RUN: llvm-objdump --syms %t/nolto-nolto.out | FileCheck %s

; (2) NoLTO-ThinLTO
; RUN: %lld -dylib -lSystem -L%t %t/f.nolto.o %t/b.thinlto.o -o %t/nolto-thinlto.out
; RUN: llvm-objdump --syms %t/nolto-thinlto.out | FileCheck %s

; (3) ThinLTO-NoLTO
; RUN: %lld -dylib -lSystem -L%t %t/f.thinlto.o %t/b.nolto.o -o %t/thinlto-nolto.out
; RUN: llvm-objdump --syms %t/thinlto-nolto.out | FileCheck %s

; (4) NoLTO-LTO
; RUN: %lld -dylib -lSystem -L%t %t/f.nolto.o %t/b.lto.o -o %t/nolto-lto.out
; RUN: llvm-objdump --syms %t/nolto-lto.out | FileCheck %s

; (5) LTO-NoLTO
; RUN: %lld -dylib -lSystem -L%t %t/f.lto.o %t/b.nolto.o -o %t/lto-nolto.out
; RUN: llvm-objdump --syms %t/lto-nolto.out | FileCheck %s

; (6) LTO-ThinLTO
; RUN: %lld -dylib -lSystem -L%t %t/f.lto.o %t/b.thinlto.o -o %t/lto-thinlto.out
; RUN: llvm-objdump --syms %t/lto-thinlto.out | FileCheck %s

; (7) ThinLTO-NoLTO
; RUN: %lld -dylib -lSystem -L%t %t/f.thinlto.o %t/b.lto.o -o %t/thinlto-lto.out
; RUN: llvm-objdump --syms %t/thinlto-lto.out | FileCheck %s

; We expect to resolve _weak1 from f.ll and _weak2 from b.ll as per the input order.
; As _weak2 from q.ll pulled in via LC_LINKER_OPTION is processed
; in the second pass, it won't prevail due to _weak2 from b.ll.

; CHECK:          w    O __TEXT,f _weak1
; CHECK:          w    O __TEXT,b _weak2

;--- q.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define i32 @weak2() section "__TEXT,q" {
  ret i32 2
}

;--- f.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

!0 = !{!"-lq"}
!llvm.linker.options = !{!0}

define weak i32 @weak1() section "__TEXT,f" {
  %call = call i32 @weak2()
  %add = add nsw i32 %call, 1
  ret i32 %add
}

declare i32 @weak2(...)

;--- b.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define weak i32 @weak1() section "__TEXT,b" {
  ret i32 3
}

define weak i32 @weak2() section "__TEXT,b" {
  ret i32 4
}
