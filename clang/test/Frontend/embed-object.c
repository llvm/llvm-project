// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -emit-llvm -fembed-offload-object=%S/Inputs/empty.h -o - %s | FileCheck %s

// CHECK: @[[OBJECT:.+]] = private constant [0 x i8] zeroinitializer, section ".llvm.offloading", align 8
// CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @[[OBJECT]]], section "llvm.metadata"

void foo(void) {}

// CHECK: !llvm.embedded.objects = !{![[METADATA:[0-9]+]]}
// CHECK: ![[METADATA]] = !{ptr @[[OBJECT]], !".llvm.offloading"}
