// Check that clang emits the "target-abi" module flag for LoongArch using the
// target ABI string.

// Default ABIs (no -target-abi override).
// RUN: %clang_cc1 -triple loongarch32 -emit-llvm -o - %s | FileCheck --check-prefix=ILP32D %s
// RUN: %clang_cc1 -triple loongarch64 -emit-llvm -o - %s | FileCheck --check-prefix=LP64D %s

// Explicit -target-abi overrides differing from the triple default.
// RUN: %clang_cc1 -triple loongarch32 -target-abi ilp32f -emit-llvm -o - %s | FileCheck --check-prefix=ILP32F %s
// RUN: %clang_cc1 -triple loongarch32 -target-abi ilp32s -emit-llvm -o - %s | FileCheck --check-prefix=ILP32S %s
// RUN: %clang_cc1 -triple loongarch64 -target-abi lp64f -emit-llvm -o - %s | FileCheck --check-prefix=LP64F %s
// RUN: %clang_cc1 -triple loongarch64 -target-abi lp64s -emit-llvm -o - %s | FileCheck --check-prefix=LP64S %s

// ILP32D: !{i32 1, !"target-abi", !"ilp32d"}
// ILP32F: !{i32 1, !"target-abi", !"ilp32f"}
// ILP32S: !{i32 1, !"target-abi", !"ilp32s"}
// LP64D: !{i32 1, !"target-abi", !"lp64d"}
// LP64F: !{i32 1, !"target-abi", !"lp64f"}
// LP64S: !{i32 1, !"target-abi", !"lp64s"}

int x;
