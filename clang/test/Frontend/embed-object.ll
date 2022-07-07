; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK

; CHECK: @[[OBJECT_1:.+]] = private constant [0 x i8] zeroinitializer, section ".llvm.offloading", align 8
; CHECK: @[[OBJECT_2:.+]] = private constant [0 x i8] zeroinitializer, section ".llvm.offloading", align 8
; CHECK: @llvm.compiler.used = appending global [3 x ptr] [ptr @x, ptr @[[OBJECT_1]], ptr @[[OBJECT_2]]], section "llvm.metadata"

@x = private constant i8 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @x], section "llvm.metadata"

define i32 @foo() {
  ret i32 0
}

; CHECK: !llvm.embedded.objects = !{![[METADATA_1:[0-9]+]], ![[METADATA_2:[0-9]+]]}
; CHECK: ![[METADATA_1]] = !{ptr @[[OBJECT_1]], !".llvm.offloading"}
; CHECK: ![[METADATA_2]] = !{ptr @[[OBJECT_2]], !".llvm.offloading"}
