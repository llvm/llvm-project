// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-auth-traps \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini \
// RUN:   -fptrauth-init-fini-address-discrimination \
// RUN:   -fptrauth-elf-got \
// RUN:   -fptrauth-indirect-gotos \
// RUN:   -fptrauth-type-info-vtable-pointer-discrimination \
// RUN:   -fptrauth-function-pointer-type-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=ALL

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-intrinsics %s | FileCheck %s --check-prefix=INTRIN

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-calls %s | FileCheck %s --check-prefix=CALL

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-returns %s | FileCheck %s --check-prefix=RET

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-auth-traps %s | FileCheck %s --check-prefix=TRAP

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-vtable-pointer-address-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=VPTRADDR

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=VPTRTYPE

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-init-fini %s | \
// RUN:   FileCheck %s --check-prefix=INITFINI

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-init-fini -fptrauth-init-fini-address-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=INITFINIADDR

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-elf-got %s | FileCheck %s --check-prefix=ELFGOT

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-indirect-gotos %s | FileCheck %s --check-prefix=GOTOS

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-type-info-vtable-pointer-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=TYPEINFO

// RUN: %clang_cc1 -triple aarch64-linux -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-function-pointer-type-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=FPTRTYPE

// ALL: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// ALL: !{i32 1, !"aarch64-elf-pauthabi-version", i32 4095}

// INTRIN: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// INTRIN: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1}

// CALL: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// CALL: !{i32 1, !"aarch64-elf-pauthabi-version", i32 2}

// RET: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// RET: !{i32 1, !"aarch64-elf-pauthabi-version", i32 4}

// TRAP: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// TRAP: !{i32 1, !"aarch64-elf-pauthabi-version", i32 8}

// VPTRADDR: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// VPTRADDR: !{i32 1, !"aarch64-elf-pauthabi-version", i32 18}

// VPTRTYPE: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// VPTRTYPE: !{i32 1, !"aarch64-elf-pauthabi-version", i32 34}

// INITFINI: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// INITFINI: !{i32 1, !"aarch64-elf-pauthabi-version", i32 66}

// INITFINIADDR: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// INITFINIADDR: !{i32 1, !"aarch64-elf-pauthabi-version", i32 194}

// ELFGOT: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// ELFGOT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 256}

// GOTOS: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// GOTOS: !{i32 1, !"aarch64-elf-pauthabi-version", i32 512}

// TYPEINFO: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// TYPEINFO: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1026}

// FPTRTYPE: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// FPTRTYPE: !{i32 1, !"aarch64-elf-pauthabi-version", i32 2050}

void foo() {}
