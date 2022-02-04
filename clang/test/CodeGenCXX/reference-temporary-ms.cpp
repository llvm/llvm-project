// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -fms-extensions | FileCheck %s

const int __declspec(dllexport) &Exported = 42;

// The reference temporary shouldn't be dllexport, even if the reference is.
// CHECK: @"?$RT1@Exported@@3ABHB" = internal constant i32 42

// CHECK: @"?Exported@@3ABHB" = dso_local dllexport constant i32* @"?$RT1@Exported@@3ABHB"
