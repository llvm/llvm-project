// RUN: %clang_cc1 -no-opaque-pointers -triple i686-apple-ios10.3 -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK32
// RUN: %clang_cc1 -no-opaque-pointers -triple i686--windows-msvc -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK32
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-ios10.3 -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK64
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64--windows-msvc -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK64

void f(id a) {
  for (id i in a)
    (void)i;
}

// CHECK32: call i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i32)*)
// CHECK32: call i32 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i32)*)

// CHECK64: call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)
// CHECK64: call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*, %struct.__objcFastEnumerationState*, [16 x i8*]*, i64)*)

