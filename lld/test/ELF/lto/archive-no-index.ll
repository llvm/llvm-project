; REQUIRES: x86
; Tests that we suggest that LTO symbols missing from an archive index
; may be the cause of undefined references, but only if we both
; encountered an empty archive index and undefined references (to prevent
; noisy false alarms).

; RUN: rm -fr %T/archive-no-index
; RUN: mkdir %T/archive-no-index
; RUN: llvm-as %S/Inputs/archive.ll -o %T/archive-no-index/f.o
; RUN: llvm-ar cr %T/archive-no-index/libf.a
; RUN: llvm-ar qS %T/archive-no-index/libf.a %T/archive-no-index/f.o
; RUN: llvm-as %s -o %t.o
; RUN: not ld.lld -emain -m elf_x86_64 %t.o -o %t %T/archive-no-index/libf.a \
; RUN:     2>&1 | FileCheck --check-prefix=NOTE %s

; RUN: llvm-ar crs %T/archive-no-index/libfs.a %T/archive-no-index/f.o
; RUN: ld.lld -emain -m elf_x86_64 %t.o -o %t %T/archive-no-index/libf.a \
; RUN:     %T/archive-no-index/libfs.a

; RUN: llvm-as %S/Inputs/archive-3.ll -o %T/archive-no-index/foo.o
; RUN: llvm-ar crs %T/archive-no-index/libfoo.a %T/archive-no-index/foo.o
; RUN: not ld.lld -emain -m elf_x86_64 %t.o -o %t %T/archive-no-index/libfoo.a \
; RUN:     2>&1 | FileCheck --check-prefix=NO-NOTE %s

; NOTE: undefined symbol: f
; NOTE: archive listed no symbols

; NO-NOTE: undefined symbol: f
; NO-NOTE-NOT: archive listed no symbols

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f()

define i32 @main() {
  call void @f()
  ret i32 0
}
