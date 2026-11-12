// RUN: %clang_cc1 -triple riscv32 -emit-llvm -o - %s | FileCheck -check-prefix=EMPTY-ILP32 %s
// RUN: %clang_cc1 -triple riscv32 -emit-llvm -target-feature +f -target-feature +d -o - %s | FileCheck -check-prefix=EMPTY-ILP32D %s
// RUN: %clang_cc1 -triple riscv32 -target-abi ilp32 -emit-llvm -o - %s | FileCheck -check-prefix=ILP32 %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +f -target-abi ilp32f -emit-llvm -o - %s | FileCheck -check-prefix=ILP32F %s
// RUN: %clang_cc1 -triple riscv32 -target-feature +d -target-feature +f -target-abi ilp32d -emit-llvm -o - %s | FileCheck -check-prefix=ILP32D %s
// RUN: %clang_cc1 -triple riscv64 -emit-llvm -o - %s | FileCheck -check-prefix=EMPTY-LP64 %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d -emit-llvm -o - %s | FileCheck -check-prefix=EMPTY-LP64D %s
// RUN: %clang_cc1 -triple riscv64 -target-abi lp64 -emit-llvm -o - %s | FileCheck -check-prefix=LP64 %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-abi lp64f -emit-llvm -o - %s | FileCheck -check-prefix=LP64F %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +d -target-feature +f -target-abi lp64d -emit-llvm -o - %s | FileCheck -check-prefix=LP64D %s

// Test expected behavior when giving -target-cpu
// This cc1 test is similar to clang with -march=rv32ifd -mcpu=sifive-e31, default abi is ilp32d
// RUN: %clang_cc1 -triple riscv32 -emit-llvm -target-feature +f -target-feature +d -target-cpu sifive-e31 -o - %s | FileCheck -check-prefix=EMPTY-ILP32D %s
// This cc1 test is similar to clang with -march=rv64i -mcpu=sifive-u74, default abi is lp64
// RUN: %clang_cc1 -triple riscv64 -emit-llvm -o - -target-cpu sifive-u74 %s | FileCheck -check-prefix=EMPTY-LP64 %s

// EMPTY-ILP32: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32"}
// EMPTY-ILP32D: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32d"}
// ILP32: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32"}
// ILP32F: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32f"}
// ILP32D: !{{[0-9]+}} = !{i32 1, !"target-abi", !"ilp32d"}

// EMPTY-LP64: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64"}
// EMPTY-LP64D: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64d"}
// LP64: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64"}
// LP64F: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64f"}
// LP64D: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64d"}
