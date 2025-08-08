// UNSUPPORTED: system-windows


// RUN: %clang --target=armv6m-none-eabi --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree -### 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-none-elf --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree -### 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf --gcc-toolchain=%S/Inputs/basic_riscv32_tree -### 2>&1 | FileCheck --check-prefix=NOCHECK %s
// RUN: %clang --target=riscv64-unknown-elf --gcc-toolchain=%S/Inputs/basic_riscv64_tree -### 2>&1 | FileCheck --check-prefix=NOCHECK %s

// CHECK: warning: no multilib structure encoded for Arm, Aarch64 and PPC targets
// NOCHECK-NOT: warning: no multilib structure encoded for Arm, Aarch64 and PPC targets
