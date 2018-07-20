// Check gnutools are invoked with propagated values for -mabi and -march.

// RUN: %clang -target riscv32 -fno-integrated-as %s -###  -c \
// RUN: 2>&1 | FileCheck -check-prefix=MABI-ILP32 %s

// RUN: %clang -target riscv32 -fno-integrated-as -march=rv32g %s -### -c \
// RUN: 2>&1 | FileCheck -check-prefix=MABI-ILP32-MARCH-G %s

// RUN: %clang -target riscv64 -fno-integrated-as %s -###  -c \
// RUN: 2>&1 | FileCheck -check-prefix=MABI-ILP64 %s

// RUN: %clang -target riscv64 -fno-integrated-as -march=rv64g %s -### -c \
// RUN: 2>&1 | FileCheck -check-prefix=MABI-ILP64-MARCH-G %s

// MABI-ILP32: "{{.*}}as{{(.exe)?}}" "-mabi" "ilp32"
// MABI-ILP32-MARCH-G: "{{.*}}as{{(.exe)?}}" "-mabi" "ilp32" "-march" "rv32g"

// MABI-ILP64: "{{.*}}as{{(.exe)?}}" "-mabi" "lp64"
// MABI-ILP64-MARCH-G: "{{.*}}as{{(.exe)?}}" "-mabi" "lp64" "-march" "rv64g"
