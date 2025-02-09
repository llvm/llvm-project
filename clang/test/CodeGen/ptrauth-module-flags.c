// RUN: %clang_cc1 -triple aarch64-linux-gnu                   -emit-llvm %s  -o - | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-elf-got -emit-llvm %s  -o - | FileCheck %s --check-prefix=ELFGOT
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls   -emit-llvm %s  -o - | FileCheck %s --check-prefix=PERSONALITY

// ELFGOT:      !llvm.module.flags = !{
// ELFGOT-SAME: !1
// ELFGOT:      !1 = !{i32 8, !"ptrauth-elf-got", i32 1}

// PERSONALITY:      !llvm.module.flags = !{
// PERSONALITY-SAME: !1
// PERSONALITY:      !1 = !{i32 8, !"ptrauth-sign-personality", i32 1}

// OFF-NOT: "ptrauth-
