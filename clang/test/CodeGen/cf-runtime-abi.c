// RUN: %clang_cc1 -triple x86_64-apple-macosx -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC-LLP64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=objc -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=objc -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC-LLP64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcf-runtime-abi=objc -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=standalone -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=standalone -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC-LLP64
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcf-runtime-abi=standalone -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-OBJC

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-32

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift-5.0 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift-5.0 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift-5.0 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift-5.0 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-5_0-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift-5.0 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift-5.0 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-5_0-32

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift-4.2 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift-4.2 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift-4.2 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift-4.2 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_2-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift-4.2 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_2-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift-4.2 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_2-32

// RUN: %clang_cc1 -triple x86_64-apple-macosx -fcf-runtime-abi=swift-4.1 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-64
// RUN: %clang_cc1 -triple aarch64-apple-ios -fcf-runtime-abi=swift-4.1 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-64
// RUN: %clang_cc1 -triple armv7k-apple-watchos -fcf-runtime-abi=swift-4.1 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-32
// RUN: %clang_cc1 -triple armv7-apple-tvos -fcf-runtime-abi=swift-4.1 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-DARWIN-4_1-32
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fcf-runtime-abi=swift-4.1 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_1-64
// RUN: %clang_cc1 -triple armv7-unknown-linux-android -fcf-runtime-abi=swift-4.1 -S -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-SWIFT-4_1-32

const __NSConstantString *s = __builtin___CFStringMakeConstantString("");

// CHECK-OBJC: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i64 0 }
// CHECK-OBJC-LLP64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }

// CHECK-SWIFT-DARWIN-5_0-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (i64* @"$s15SwiftFoundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i64 0 }
// CHECK-SWIFT-DARWIN-5_0-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (i32* @"$s15SwiftFoundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-5_0-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (i64* @"$s10Foundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i64 0 }
// CHECK-SWIFT-5_0-64-SAME: align 8
// CHECK-SWIFT-5_0-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (i32* @"$s10Foundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-5_0-32-SAME: align 8

// CHECK-SWIFT-DARWIN-4_2-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (i64* @"$S15SwiftFoundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-DARWIN-4_2-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (i32* @"$S15SwiftFoundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-4_2-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (i64* @"$S10Foundation19_NSCFConstantStringCN" to i64), i64 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-4_2-64-SAME: align 8
// CHECK-SWIFT-4_2-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (i32* @"$S10Foundation19_NSCFConstantStringCN" to i32), i32 1, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-4_2-32-SAME: align 8

// CHECK-SWIFT-DARWIN-4_1-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (i64* @__T015SwiftFoundation19_NSCFConstantStringCN to i64), i64 5, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-DARWIN-4_1-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (i32* @__T015SwiftFoundation19_NSCFConstantStringCN to i32), i32 5, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-4_1-64: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i64 ptrtoint (i64* @__T010Foundation19_NSCFConstantStringCN to i64), i64 5, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-4_1-64-SAME: align 8
// CHECK-SWIFT-4_1-32: @_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32 ptrtoint (i32* @__T010Foundation19_NSCFConstantStringCN to i32), i32 5, i64 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i32 0, i32 0), i32 0 }
// CHECK-SWIFT-4_1-32-SAME: align 8

