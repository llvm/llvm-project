// RUN: %clang_cc1 -triple x86_64-apple-macos11 -darwin-target-variant-triple x86_64-apple-ios14-macabi -target-sdk-version=11.1 -darwin-target-variant-sdk-version=14.1 -emit-llvm -o - %s | FileCheck %s

// CHECK: !llvm.module.flags = !{!0, !1, !2
// CHECK: !0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 1]}
// CHECK: !1 = !{i32 2, !"darwin.target_variant.triple", !"x86_64-apple-ios14-macabi"}
// CHECK: !2 = !{i32 2, !"darwin.target_variant.SDK Version", [2 x i32] [i32 14, i32 1]}
