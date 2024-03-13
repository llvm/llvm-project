// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-intrinsics \
// RUN:   -fptrauth-calls \
// RUN:   -fptrauth-returns \
// RUN:   -fptrauth-auth-traps \
// RUN:   -fptrauth-vtable-pointer-address-discrimination \
// RUN:   -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-init-fini %s | \
// RUN:   FileCheck %s --check-prefix=ALL

// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-intrinsics %s | FileCheck %s --check-prefix=INTRIN

// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-calls %s | FileCheck %s --check-prefix=CALL

// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-returns %s | FileCheck %s --check-prefix=RET

// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-auth-traps %s | FileCheck %s --check-prefix=TRAP

// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-vtable-pointer-address-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=VPTRADDR

// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination %s | \
// RUN:   FileCheck %s --check-prefix=VPTRTYPE

// RUN: %clang -target aarch64-linux -S -emit-llvm -o - \
// RUN:   -fptrauth-calls -fptrauth-init-fini %s | \
// RUN:   FileCheck %s --check-prefix=INITFINI

// REQUIRES: aarch64-registered-target

// ALL: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// ALL: !{i32 1, !"aarch64-elf-pauthabi-version", i32 127}

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

void foo() {}
