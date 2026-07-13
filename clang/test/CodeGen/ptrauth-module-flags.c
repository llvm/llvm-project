// RUN: %clang_cc1 -triple aarch64-linux-gnu                   -emit-llvm %s  -o - | FileCheck %s --check-prefix=OFF-LINUX
// RUN: %clang_cc1 -triple aarch64-freebsd                     -emit-llvm %s  -o - | FileCheck %s --check-prefix=OFF-ELF
// RUN: %clang_cc1 -triple aarch64-darwin                      -emit-llvm %s  -o - | FileCheck %s --check-prefix=ABSENT
// RUN: %clang_cc1 -triple aarch64-windows                     -emit-llvm %s  -o - | FileCheck %s --check-prefix=ABSENT
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

// OFF-LINUX:      !llvm.module.flags = !{
// OFF-LINUX-SAME: !0
// OFF-LINUX-SAME: !1
// OFF-LINUX:      !0 = !{i32 1, !"ptrauth-elf-got", i32 0}
// OFF-LINUX:      !1 = !{i32 1, !"ptrauth-sign-personality", i32 0}

// OFF-ELF:      !llvm.module.flags = !{
// OFF-ELF-SAME: !0
// OFF-ELF:      !0 = !{i32 1, !"ptrauth-elf-got", i32 0}
// OFF-ELF-NOT:  ptrauth-sign-personality

// ABSENT-NOT: "ptrauth-
