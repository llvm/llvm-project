// ; Check that -flto=thin without -fsplit-lto-unit has EnableSplitLTOUnit = 0
// RUN: %clang_cc1 -flto=thin -emit-llvm-bc < %s | llvm-dis -o - | FileCheck %s
// CHECK: !{i32 1, !"EnableSplitLTOUnit", i32 0}
//
// ; Check that -flto=thin with -fsplit-lto-unit has EnableSplitLTOUnit = 1
// RUN: %clang_cc1 -flto=thin -fsplit-lto-unit -emit-llvm-bc < %s | llvm-dis -o - | FileCheck %s --check-prefix=SPLIT
// SPLIT: !{i32 1, !"EnableSplitLTOUnit", i32 1}
//
// ; Check that regular LTO has EnableSplitLTOUnit = 1
// RUN: %clang_cc1 -flto -triple x86_64-pc-linux-gnu -emit-llvm-bc < %s | llvm-dis -o - | FileCheck %s --implicit-check-not="EnableSplitLTOUnit" --check-prefix=SPLIT

int main(void) {}
