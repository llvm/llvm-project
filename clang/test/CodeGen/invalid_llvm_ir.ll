; RUN: not %clang %s 2>&1 | FileCheck %s
; RUN: llvm-as -disable-verify < %s > %t.bc
; RUN: not %clang %t.bc 2>&1 | FileCheck %s

; CHECK: error: invalid LLVM IR input: PHINode should have one entry for each predecessor of its parent basic block!
; CHECK-NEXT: %phi = phi i32 [ 0, %entry ]

define void @test() {
entry:
  %phi = phi i32 [ 0, %entry ]
  ret void
}
