; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h,section1 \
; RUN:    -fembed-offload-object=%S/Inputs/empty.h,section2 -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK

; CHECK: @[[OBJECT1:.+]] = hidden constant [0 x i8] zeroinitializer, section ".llvm.offloading.section1"
; CHECK: @[[OBJECT2:.+]] = hidden constant [0 x i8] zeroinitializer, section ".llvm.offloading.section2"
; CHECK: @llvm.compiler.used = appending global [3 x i8*] [i8* @x, i8* getelementptr inbounds ([0 x i8], [0 x i8]* @[[OBJECT1]], i32 0, i32 0), i8* getelementptr inbounds ([0 x i8], [0 x i8]* @[[OBJECT2]], i32 0, i32 0)], section "llvm.metadata"

@x = private constant i8 1
@llvm.compiler.used = appending global [1 x i8*] [i8* @x], section "llvm.metadata"

define i32 @foo() {
  ret i32 0
}
