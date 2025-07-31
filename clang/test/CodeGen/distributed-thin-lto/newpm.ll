; FIXME: This test should use CHECK-NEXT to keep up-to-date.
; REQUIRES: x86-registered-target

;; Validate that we set up the ThinLTO post link pipeline at O2 and O3
;; for a ThinLTO distributed backend invoked via clang.
;; Since LLVM tests already more thoroughly test this pipeline, and to
;; avoid making this clang test too sensitive to LLVM pipeline changes,
;; here we simply confirm that an LTO backend-specific pass is getting
;; invoked (WPD).

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,main,px

; RUN: %clang -target x86_64-grtev4-linux-gnu \
; RUN:   -O2 -Xclang -fdebug-pass-manager \
; RUN:   -c -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o 2>&1 | FileCheck -check-prefix=CHECK-O %s --dump-input=fail

; RUN: %clang -target x86_64-grtev4-linux-gnu \
; RUN:   -O3 -Xclang -fdebug-pass-manager \
; RUN:   -c -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o 2>&1 | FileCheck -check-prefixes=CHECK-O %s --dump-input=fail

; CHECK-O: Running pass: WholeProgramDevirtPass

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define i32 @main() {
  br label %b
b:
  br label %b
  ret i32 0
}
