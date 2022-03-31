; REQUIRES: asserts

; RUN: llvm-as -o %t.bc %s

;; Try to save statistics to file.
; RUN: ld.lld --stats-file=%t2.stats -m elf_x86_64 -r -o %t.o %t.bc
; RUN: FileCheck --input-file=%t2.stats %s

; CHECK: {
; CHECK: "asm-printer.EmittedInsts":
; CHECK: "inline.NumInlined":
; CHECK: "prologepilog.NumFuncSeen":
; CHECK: }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @patatino()

define i32 @tinkywinky() {
  %a = call i32 @patatino()
  ret i32 %a
}

define i32 @main() !prof !0 {
  %i = call i32 @tinkywinky()
  ret i32 %i
}

!0 = !{!"function_entry_count", i64 300}
