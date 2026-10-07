// Check that clang emits the "float-abi" module flag only when the resolved
// floating-point ABI differs from the target default.

// Default (soft) ABI on a soft-default triple: no flag.
// RUN: %clang_cc1 -triple arm-none-none-eabi -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=NONE

// Explicit hard on a soft-default triple: flag emitted.
// RUN: %clang_cc1 -triple arm-none-none-eabi -mfloat-abi hard -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=HARD

// Explicit soft on a soft-default triple: matches default, no flag.
// RUN: %clang_cc1 -triple arm-none-none-eabi -mfloat-abi soft -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=NONE

// Default (hard) ABI on a hard-default triple: no flag.
// RUN: %clang_cc1 -triple arm-none-none-eabihf -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=NONE

// Explicit soft on a hard-default triple: flag emitted.
// RUN: %clang_cc1 -triple arm-none-none-eabihf -mfloat-abi soft -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=SOFT

// softfp collapses to soft: on a hard-default triple, flag emitted as soft.
// RUN: %clang_cc1 -triple arm-none-none-eabihf -mfloat-abi softfp -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=SOFT

void f(void) {}

// NONE-NOT: !"float-abi"
// HARD: !{i32 1, !"float-abi", !"hard"}
// SOFT: !{i32 1, !"float-abi", !"soft"}
