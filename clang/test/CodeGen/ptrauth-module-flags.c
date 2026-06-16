// RUN: %clang_cc1 -triple aarch64-linux-gnu                   -emit-llvm %s  -o - | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-elf-got -emit-llvm %s  -o - | FileCheck %s --check-prefix=ELFGOT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls   -emit-llvm %s  -o - | FileCheck %s --check-prefix=PERSONALITY

// ELFGOT:      !llvm.module.flags = !{
// ELFGOT-SAME: !0
// ELFGOT-SAME: !1
// ELFGOT:      !0 = !{i32 1, !"ptrauth-elf-got", i32 1}
// ELFGOT:      !1 = !{i32 1, !"ptrauth-sign-personality", i32 0}

// PERSONALITY:      !llvm.module.flags = !{
// PERSONALITY-SAME: !0
// PERSONALITY-SAME: !1
// PERSONALITY:      !0 = !{i32 1, !"ptrauth-elf-got", i32 0}
// PERSONALITY:      !1 = !{i32 1, !"ptrauth-sign-personality", i32 1}

// OFF:      !llvm.module.flags = !{
// OFF-SAME: !0
// OFF-SAME: !1
// OFF:      !0 = !{i32 1, !"ptrauth-elf-got", i32 0}
// OFF:      !1 = !{i32 1, !"ptrauth-sign-personality", i32 0}
