// RUN: %clang_cc1 -triple i686-apple-ios10.3 -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK32
// RUN: %clang_cc1 -triple i686--windows-msvc -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK32
// RUN: %clang_cc1 -triple x86_64-apple-ios10.3 -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK64
// RUN: %clang_cc1 -triple x86_64--windows-msvc -fobjc-runtime=ios-6.0 -Os -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK64

void f(id a) {
  for (id i in a)
    (void)i;
}

// CHECK32: call i32 @objc_msgSend
// CHECK32: call i32 @objc_msgSend

// CHECK64: call i64 @objc_msgSend
// CHECK64: call i64 @objc_msgSend

