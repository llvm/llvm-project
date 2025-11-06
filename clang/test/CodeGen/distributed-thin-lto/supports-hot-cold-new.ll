; REQUIRES: x86-registered-target

;; Test that passing -supports-hot-cold-new to the thin link prevents memprof
;; metadata and attributes from being removed from the distributed ThinLTO
;; backend, and vice versa without passing -supports-hot-cold-new.

;; First check with -supports-hot-cold-new.
; RUN: opt -thinlto-bc %s >%t.o
; RUN: llvm-lto2 run %t.o -save-temps \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -o %t.out

;; Ensure that the index file reflects the -supports-hot-cold-new, as that is
;; how the ThinLTO backend behavior is controlled.
; RUN: llvm-dis %t.out.index.bc -o - | FileCheck %s --check-prefix=CHECK-INDEX-ON
;; Flags are printed in decimal, but this corresponds to 0x161, and 0x100 is
;; the value indicating -supports-hot-cold-new was enabled.
; CHECK-INDEX-ON: flags: 353

; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t1.o -x ir %t.o -c -fthinlto-index=%t.o.thinlto.bc -save-temps=obj

; RUN: llvm-dis %t.s.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR
; CHECK-IR: !memprof {{.*}} !callsite
; CHECK-IR: "memprof"="cold"

;; Next check without -supports-hot-cold-new, we should not perform
;; context disambiguation, and we should strip memprof metadata and
;; attributes before optimization during the distributed backend.
; RUN: llvm-lto2 run %t.o -save-temps \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -o %t.out

;; Ensure that the index file reflects not having -supports-hot-cold-new.
; RUN: llvm-dis %t.out.index.bc -o - | FileCheck %s --check-prefix=CHECK-INDEX-OFF
;; Flags are printed in decimal, but this corresponds to 0x61, without 0x100 set.
; CHECK-INDEX-OFF: flags: 97

; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t1.o -x ir %t.o -c -fthinlto-index=%t.o.thinlto.bc -save-temps=obj

; RUN: llvm-dis %t.s.4.opt.bc -o - | FileCheck %s \
; RUN: --implicit-check-not "!memprof" --implicit-check-not "!callsite" \
; RUN: --implicit-check-not "memprof"="cold"

;; Ensure the attributes and metadata are stripped when running a non-LTO pipeline.
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -x ir %t.o -S -emit-llvm -o - | FileCheck %s \
; RUN: 	--implicit-check-not "!memprof" --implicit-check-not "!callsite" \
; RUN: 	--implicit-check-not "memprof"="cold"

source_filename = "thinlto-distributed-supports-hot-cold-new.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !0, !callsite !5
  %call1 = call ptr @_Znam(i64 0) #1
  ret i32 0
}

declare ptr @_Znam(i64)

attributes #0 = { noinline optnone }
attributes #1 = { "memprof"="cold" }

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 9086428284934609951, i64 -5964873800580613432, i64 2732490490862098848, i64 -3421689549917153178}
!5 = !{i64 9086428284934609951}
