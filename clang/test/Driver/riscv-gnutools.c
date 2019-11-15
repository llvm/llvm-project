// Check gnutools are invoked with propagated values for -mabi and -march.
//
// This test also checks the default -march/-mabi for certain targets.

// 32-bit checks

// Check default on riscv32-unknown-elf
// RUN: %clang -target riscv32-unknown-elf -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck -check-prefix=CHECK-RV32IMAC-ILP32 %s

// Check default on riscv32-unknown-linux-gnu
// RUN: %clang -target riscv32-unknown-linux-gnu -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck -check-prefix=CHECK-RV32IMAFDC-ILP32D %s

// Check default when -march=rv32g specified
// RUN: %clang -target riscv32 -fno-integrated-as %s -### -c -march=rv32g \
// RUN: 2>&1 | FileCheck -check-prefix=CHECK-RV32G-ILP32D %s

// CHECK-RV32IMAC-ILP32: "{{.*}}as{{(.exe)?}}" "-mabi" "ilp32" "-march" "rv32imac"
// CHECK-RV32IMAFDC-ILP32D: "{{.*}}as{{(.exe)?}}" "-mabi" "ilp32d" "-march" "rv32imafdc"
// CHECK-RV32G-ILP32D: "{{.*}}as{{(.exe)?}}" "-mabi" "ilp32d" "-march" "rv32g"


// 64-bit checks

// Check default on riscv64-unknown-elf
// RUN: %clang -target riscv64-unknown-elf -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck -check-prefix=CHECK-RV64IMAC-LP64 %s

// Check default on riscv64-unknown-linux-gnu
// RUN: %clang -target riscv64-unknown-linux-gnu -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck -check-prefix=CHECK-RV64IMAFDC-LP64D %s

// Check default when -march=rv64g specified
// RUN: %clang -target riscv64 -fno-integrated-as %s -### -c -march=rv64g \
// RUN: 2>&1 | FileCheck -check-prefix=CHECK-RV64G-LP64D %s

// CHECK-RV64IMAC-LP64: "{{.*}}as{{(.exe)?}}" "-mabi" "lp64" "-march" "rv64imac"
// CHECK-RV64IMAFDC-LP64D: "{{.*}}as{{(.exe)?}}" "-mabi" "lp64d" "-march" "rv64imafdc"
// CHECK-RV64G-LP64D: "{{.*}}as{{(.exe)?}}" "-mabi" "lp64d" "-march" "rv64g"
