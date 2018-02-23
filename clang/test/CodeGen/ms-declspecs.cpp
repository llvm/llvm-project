// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -fms-compatibility -o - | FileCheck %s

// selectany turns extern "C" variable declarations into definitions.
extern __declspec(selectany) int x1;
extern "C" __declspec(selectany) int x2;
extern "C++" __declspec(selectany) int x3;
extern "C" {
__declspec(selectany) int x4;
}
__declspec(selectany) int x5;
// CHECK: @"\01?x1@@3HA" = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @x2 = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @"\01?x3@@3HA"  = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @x4 = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @"\01?x5@@3HA"  = weak_odr dso_local global i32 0, comdat, align 4
