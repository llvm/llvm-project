; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h,,, \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h,,, -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK

; CHECK: @[[OBJECT_1:.+]] = private constant [120 x i8] c"\10\FF\10\AD{{.*}}\00", section ".llvm.offloading", align 8
; CHECK: @[[OBJECT_2:.+]] = private constant [120 x i8] c"\10\FF\10\AD{{.*}}\00", section ".llvm.offloading", align 8
; CHECK: @llvm.compiler.used = appending global [3 x ptr] [ptr @x, ptr @[[OBJECT_1]], ptr @[[OBJECT_2]]], section "llvm.metadata"

@x = private constant i8 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @x], section "llvm.metadata"

define i32 @foo() {
  ret i32 0
}
