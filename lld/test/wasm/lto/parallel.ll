; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: llvm-as -o a.bc %s
; RUN: wasm-ld --lto-partitions=2 -save-temps -o out a.bc -r
; RUN: llvm-nm out.lto.o | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-nm out.lto.1.o | FileCheck --check-prefix=CHECK1 %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown-wasm"

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
