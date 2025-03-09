// check that gnu assembler is invoked with arm baremetal as well

// RUN: %clang --target=armv6m-none-eabi  --gcc-toolchain=%S/Inputs/basic_riscv32_tree -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck %s

// RUN: %clang --target=armv7-none-eabi  --gcc-toolchain=%S/Inputs/basic_riscv32_tree -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck %s

// RUN: %clang --target=aarch64-none-elf  --gcc-toolchain=%S/Inputs/basic_riscv32_tree -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck %s

// CHECK: "{{.*}}as{{(.exe)?}}"