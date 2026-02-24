// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=GNU
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC

// Verify __int256 ABI on Windows targets (both GNU and MSVC).
// On Win64, __int256 is passed/returned indirectly (pointer args, sret return).

// GNU-LABEL: define dso_local void @f_ret(ptr dead_on_unwind noalias writable sret(i256) align 16 %agg.result, ptr noundef dead_on_return %0)
// MSVC-LABEL: define dso_local void @f_ret(ptr dead_on_unwind noalias writable sret(i256) align 16 %agg.result, ptr noundef dead_on_return %0)
__int256 f_ret(__int256 a) { return a; }

// GNU-LABEL: define dso_local void @f_two(ptr dead_on_unwind noalias writable sret(i256) align 16 %agg.result, ptr noundef dead_on_return %0, ptr noundef dead_on_return %1)
// MSVC-LABEL: define dso_local void @f_two(ptr dead_on_unwind noalias writable sret(i256) align 16 %agg.result, ptr noundef dead_on_return %0, ptr noundef dead_on_return %1)
__int256 f_two(__int256 a, __int256 b) { return a + b; }

// GNU-LABEL: define dso_local i32 @f_narrow(ptr noundef dead_on_return %0)
// MSVC-LABEL: define dso_local i32 @f_narrow(ptr noundef dead_on_return %0)
int f_narrow(__int256 a) { return (int)a; }

// Mixed: small args passed in registers, __int256 via pointer
// GNU-LABEL: define dso_local void @f_mixed(ptr dead_on_unwind noalias writable sret(i256) align 16 %agg.result, i32 noundef %x, ptr noundef dead_on_return %0, i32 noundef %y)
// MSVC-LABEL: define dso_local void @f_mixed(ptr dead_on_unwind noalias writable sret(i256) align 16 %agg.result, i32 noundef %x, ptr noundef dead_on_return %0, i32 noundef %y)
__int256 f_mixed(int x, __int256 a, int y) { return a; }
