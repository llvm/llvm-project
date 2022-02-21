// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -emit-llvm -fembed-offload-object=%S/Inputs/empty.h,section

// CHECK: @[[OBJECT:.+]] = private constant [0 x i8] zeroinitializer, section ".llvm.offloading.section"
// CHECK: @llvm.compiler.used = appending global [3 x i8*] [i8* getelementptr inbounds ([0 x i8], [0 x i8]* @[[OBJECT1]]], section "llvm.metadata"

void foo(void) {}
