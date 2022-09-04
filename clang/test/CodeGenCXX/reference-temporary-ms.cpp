// RUN: %clang_cc1 %std_cxx98-14 -no-opaque-pointers -emit-llvm %s -o - -triple=i386-pc-win32 -fms-extensions | FileCheck %s --check-prefixes=CHECK,PRE17
// RUN: %clang_cc1 %std_cxx17- -no-opaque-pointers -emit-llvm %s -o - -triple=i386-pc-win32 -fms-extensions | FileCheck %s --check-prefixes=CHECK,CXX17

const int __declspec(dllexport) &Exported = 42;

// The reference temporary shouldn't be dllexport, even if the reference is.
// PRE17: @"?$RT1@Exported@@3ABHB" = internal constant i32 42
// CXX17: @"?$RT1@Exported@@3ABHB" = internal global i32 42

// CHECK: @"?Exported@@3ABHB" = dso_local dllexport constant i32* @"?$RT1@Exported@@3ABHB"
