// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14 -target-sdk-version=10.14.1 -emit-llvm -o - %s | FileCheck %s

// CHECK: !llvm.module.flags = !{!0
// CHECK: !0 = !{i32 2, !"SDK Version", [3 x i32] [i32 10, i32 14, i32 1]}
