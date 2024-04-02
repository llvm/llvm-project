// REQUIRES: aarch64-registered-target

// Hard-float, valid
// RUN: %clang --target=aarch64-none-elf                               -c %s -o /dev/null
// RUN: %clang --target=aarch64-none-elf                   -mabi=aapcs -c %s -o /dev/null
// RUN: %clang --target=aarch64-none-elf -march=armv8-r                -c %s -o /dev/null
// RUN: %clang --target=aarch64-none-elf -march=armv8-r    -mabi=aapcs -c %s -o /dev/null
// RUN: %clang --target=aarch64-none-elf -march=armv8-r+fp -mabi=aapcs -c %s -o /dev/null

// Soft-float, no FP
// RUN: %clang --target=aarch64-none-elf -march=armv8-r+nofp -mabi=aapcs-soft -c %s -o /dev/null
// RUN: %clang --target=aarch64-none-elf -mgeneral-regs-only -mabi=aapcs-soft -c %s -o /dev/null

// Soft-float, FP hardware: Rejected, to avoid having two incompatible ABIs for common targets.
// RUN: not %clang --target=aarch64-none-elf                        -mabi=aapcs-soft -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SOFT
// RUN: not %clang --target=aarch64-none-elf -march=armv8-r+fp      -mabi=aapcs-soft -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SOFT
// RUN: not %clang --target=aarch64-none-elf -march=armv8-r+nofp+fp -mabi=aapcs-soft -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-SOFT

// No FP, hard-float. This is accepted by the driver, but functions which
// require arguments or returns to be passed in FP registers will be rejected
// (tested elsewhere).
// RUN: %clang --target=aarch64-none-elf -march=armv8-r+nofp             -c %s -o /dev/null
// RUN: %clang --target=aarch64-none-elf -march=armv8-r+nofp -mabi=aapcs -c %s -o /dev/null
// RUN: %clang --target=aarch64-none-elf -mgeneral-regs-only -mabi=aapcs -c %s -o /dev/null

// INVALID-SOFT: error: 'aapcs-soft' ABI is not supported with FPU
