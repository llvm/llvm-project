// RUN: %clang --target=riscv32 -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -fcf-protection=return -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -fcf-protection=full -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -march=rv32i_zicfiss1p0 \
// RUN: -menable-experimental-extensions -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -march=rv32i_zicfiss1p0 \
// RUN: -menable-experimental-extensions -fcf-protection=return -E -dM %s \
// RUN: -o - | FileCheck --check-prefixes=SHSTK-MACRO %s

// RUN: %clang --target=riscv32 -march=rv32i_zicfiss1p0 \
// RUN: -menable-experimental-extensions -fcf-protection=full -E -dM %s -o - \
// RUN: | FileCheck --check-prefixes=SHSTK-MACRO %s

// RUN: %clang --target=riscv64 -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -fcf-protection=return -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -fcf-protection=full -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -march=rv64i_zicfiss1p0 \
// RUN: -menable-experimental-extensions -E -dM %s -o - | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -march=rv64i_zicfiss1p0 \
// RUN: -menable-experimental-extensions -fcf-protection=return -E -dM %s \
// RUN: -o - | FileCheck --check-prefixes=SHSTK-MACRO %s

// RUN: %clang --target=riscv64 -march=rv64i_zicfiss1p0 \
// RUN: -menable-experimental-extensions -fcf-protection=full -E -dM %s -o - \
// RUN: | FileCheck --check-prefixes=SHSTK-MACRO %s

// SHSTK-MACRO-NOT: __CET__
// SHSTK-MACRO: __riscv_shadow_stack 1{{$}}
// SHSTK-MACRO-NOT: __CET__
// NO-MACRO-NOT: __riscv_shadow_stack
