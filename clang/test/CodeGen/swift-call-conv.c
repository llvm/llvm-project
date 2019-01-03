// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -emit-llvm %s -o - | FileCheck %s

// REQUIRES: aarch64-registered-target,arm-registered-target,x86-registered-target

void __attribute__((__swiftcall__)) f(void) {}
// CHECK-LABEL: define dso_local swiftcc void @f()

