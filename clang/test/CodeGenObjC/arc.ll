; RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Os -emit-llvm -fobjc-arc -o - %s | FileCheck %s

declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.release(ptr)

; CHECK-LABEL: define void @test(
; CHECK-NOT: @objc_
; CHECK: }
define void @test(ptr %x, ptr %p) nounwind {
entry:
  br label %loop

loop:
  call ptr @llvm.objc.retain(ptr %x)
  %q = load i1, ptr %p
  br i1 %q, label %loop.more, label %exit

loop.more:
  call void @llvm.objc.release(ptr %x)
  br label %loop

exit:
  call void @llvm.objc.release(ptr %x)
  ret void
}
