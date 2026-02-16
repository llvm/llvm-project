// RUN: %clang_cc1 -E -dM -triple=riscv32 < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefixes=RV32 %s
// RUN: %clang_cc1 -E -dM -triple=riscv64 < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefixes=RV64 %s

// RV32: #define __GCC_CONSTRUCTIVE_SIZE 64
// RV32: #define __GCC_DESTRUCTIVE_SIZE 64

// RV64: #define __GCC_CONSTRUCTIVE_SIZE 64
// RV64: #define __GCC_DESTRUCTIVE_SIZE 64
