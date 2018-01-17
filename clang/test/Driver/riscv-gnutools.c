// Check gnutools are invoked with propagated values for -mabi and -march.

// RUN: %clang -target riscv32-linux-unknown-elf -fno-integrated-as \
// RUN: --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN: --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot %s -### \
// RUN: 2>&1 | FileCheck -check-prefix=MABI-ILP32 %s
// RUN: %clang -target riscv32-linux-unknown-elf -fno-integrated-as \
// RUN: -march=rv32g --gcc-toolchain=%S/Inputs/multilib_riscv_linux_sdk \
// RUN: --sysroot=%S/Inputs/multilib_riscv_linux_sdk/sysroot %s -### \
// RUN: 2>&1 | FileCheck -check-prefix=MABI-ILP32-MARCH-G %s

// MABI-ILP32: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}as" "-mabi" "ilp32"
// MABI-ILP32-MARCH-G: "{{.*}}/Inputs/multilib_riscv_linux_sdk/lib/gcc/riscv64-unknown-linux-gnu/7.2.0/../../../../riscv64-unknown-linux-gnu/bin{{/|\\\\}}as" "-mabi" "ilp32" "-march" "rv32g"

