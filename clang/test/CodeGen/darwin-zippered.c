// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14 -darwin-target-variant-triple x86_64-apple-ios12-macabi -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-apple-ios12-macabi -darwin-target-variant-triple x86_64-apple-macosx10.14 -emit-llvm -o - %s | FileCheck --check-prefix=INVERTED %s

// CHECK: !llvm.module.flags = !{!0
// CHECK:  !0 = !{i32 2, !"darwin.target_variant.triple", !"x86_64-apple-ios12-macabi"}

// INVERTED: !llvm.module.flags = !{!0
// INVERTED:  !0 = !{i32 2, !"darwin.target_variant.triple", !"x86_64-apple-macosx10.14"}
