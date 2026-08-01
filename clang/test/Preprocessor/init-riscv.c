// RUN: %clang_cc1 -E -dM -triple=riscv32 < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefixes=RV32 %s
// RUN: %clang_cc1 -E -dM -triple=riscv64 < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefixes=RV64 %s
// RUN: %clang_cc1 -E -dM -triple=riscv32-unknown-netbsd < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefixes=NETBSD %s
// RUN: %clang_cc1 -E -dM -triple=riscv64-unknown-netbsd < /dev/null | \
// RUN:     FileCheck -match-full-lines -check-prefixes=NETBSD %s

// RV32: #define __GCC_CONSTRUCTIVE_SIZE 64
// RV32: #define __GCC_DESTRUCTIVE_SIZE 64

// RV64: #define __GCC_CONSTRUCTIVE_SIZE 64
// RV64: #define __GCC_DESTRUCTIVE_SIZE 64

// NETBSD: #define __SIG_ATOMIC_MIN__ (-__SIG_ATOMIC_MAX__ - 1)
// NETBSD: #define __WCHAR_MIN__ (-__WCHAR_MAX__ - 1)
// NETBSD: #define __WINT_MIN__ 0U
