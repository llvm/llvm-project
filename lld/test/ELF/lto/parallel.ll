; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: llvm-as -o a.bc %s
; RUN: ld.lld --lto-partitions=2 -save-temps -o out a.bc -shared
; RUN: llvm-nm out.lto.o | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-nm out1.lto.o | FileCheck --check-prefix=CHECK1 %s

; RUN: not ld.lld --lto-partitions=0 a.bc -o /dev/null 2>&1 | FileCheck --check-prefix=INVALID %s
; INVALID: --lto-partitions: number of threads must be > 0

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK0-NOT: bar
; CHECK0: T foo
; CHECK0-NOT: bar
define void @foo() mustprogress {
  call void @bar()
  ret void
}

; CHECK1-NOT: foo
; CHECK1: T bar
; CHECK1-NOT: foo
define void @bar() mustprogress {
  call void @foo()
  ret void
}
