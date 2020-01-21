// RUN: %clang_cc1 -triple riscv32 -target-abi ilp32 -emit-llvm -o - %s | FileCheck -check-prefix=ILP32 %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +f -target-abi ilp32f -emit-llvm -o - %s | FileCheck -check-prefix=ILP32F %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +d -target-abi ilp32d -emit-llvm -o - %s | FileCheck -check-prefix=ILP32D %s
// RUN: %clang_cc1 -triple riscv64 -target-abi lp64 -emit-llvm -o - %s | FileCheck -check-prefix=LP64 %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-abi lp64f -emit-llvm -o - %s | FileCheck -check-prefix=LP64F %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +d -target-abi lp64d -emit-llvm -o - %s | FileCheck -check-prefix=LP64D %s

// ILP32: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32"}
// ILP32F: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32f"}
// ILP32D: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32d"}

// LP64: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64"}
// LP64F: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64f"}
// LP64D: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64d"}
