// RUN: %clang_cc1 -triple x86_64-apple-macosx -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC-LLP64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=objc -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=objc -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC-LLP64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcf-runtime-abi=objc -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=standalone -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=standalone -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC-LLP64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcf-runtime-abi=standalone -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-32

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift-5.0 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift-5.0 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift-5.0 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift-5.0 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift-5.0 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift-5.0 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-32

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift-4.2 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift-4.2 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift-4.2 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift-4.2 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift-4.2 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_2-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift-4.2 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_2-32

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift-4.1 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift-4.1 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift-4.1 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift-4.1 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift-4.1 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_1-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift-4.1 -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_1-32

const __NSConstantString *s = __builtin___CFStringMakeConstantString("");

// CHECK-OBJC: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str, i64 0 }
// CHECK-OBJC-LLP64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str, i32 0 }

// CHECK-SWIFT-DARWIN-5_0-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (ptr @"$s15SwiftFoundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, ptr @.str, i64 0 }
// CHECK-SWIFT-DARWIN-5_0-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (ptr @"$s15SwiftFoundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-5_0-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (ptr @"$s10Foundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, ptr @.str, i64 0 }
// CHECK-SWIFT-5_0-64-SAME: align 8
// CHECK-SWIFT-5_0-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (ptr @"$s10Foundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-5_0-32-SAME: align 8

// CHECK-SWIFT-DARWIN-4_2-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (ptr @"$S15SwiftFoundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-DARWIN-4_2-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (ptr @"$S15SwiftFoundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-4_2-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (ptr @"$S10Foundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-4_2-64-SAME: align 8
// CHECK-SWIFT-4_2-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (ptr @"$S10Foundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-4_2-32-SAME: align 8

// CHECK-SWIFT-DARWIN-4_1-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (ptr @__T015SwiftFoundation19_NSCFConstantStringCN to i64), i64 5, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-DARWIN-4_1-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (ptr @__T015SwiftFoundation19_NSCFConstantStringCN to i32), i32 5, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-4_1-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (ptr @__T010Foundation19_NSCFConstantStringCN to i64), i64 5, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-4_1-64-SAME: align 8
// CHECK-SWIFT-4_1-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (ptr @__T010Foundation19_NSCFConstantStringCN to i32), i32 5, i64 1992, ptr @.str, i32 0 }
// CHECK-SWIFT-4_1-32-SAME: align 8

