// RUN: not %clang_cc1 -triple riscv32 -target-feature +d -emit-llvm -target-abi ilp32e %s 2>&1 \
// RUN:     | FileCheck -check-prefix=ILP32E-WITH-FD %s

// ILP32E-WITH-FD: error: invalid feature combination: ILP32E cannot be used with the D ISA extension
