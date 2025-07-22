// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=full \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -E -dM %s -o - 2>&1 | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM -emit-llvm %s -o - | \
// RUN: FileCheck --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv32 -fcf-protection=full \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM -emit-llvm %s -o - | \
// RUN: FileCheck --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv32 -E -dM %s -o - 2>&1 | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=full \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -E -dM %s -o - 2>&1 | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv64 -fcf-protection=full \
// RUN: -mcf-branch-label-scheme=unlabeled -E -dM %s -o - | FileCheck \
// RUN: --check-prefixes=LPAD-MACRO,UNLABELED-MACRO %s

// RUN: %clang --target=riscv64 -E -dM %s -o - 2>&1 | \
// RUN: FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=unlabeled -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=func-sig -E -dM %s \
// RUN: -o - 2>&1 | FileCheck --check-prefixes=NO-MACRO %s

// LPAD-MACRO: __riscv_landing_pad 1{{$}}
// UNLABELED-MACRO: __riscv_landing_pad_unlabeled 1{{$}}
// NO-MACRO-NOT: __riscv_landing_pad
// NO-MACRO-NOT: __riscv_landing_pad_unlabeled
// NO-MACRO-NOT: __riscv_landing_pad_func_sig
