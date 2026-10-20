// check that gnu assembler is invoked with arm baremetal as well

// RUN: %clang --target=armv6m-none-eabi  --gcc-toolchain=%S/Inputs/basic_arm_gcc_tree -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck %s

// CHECK: "{{.*}}as{{(.exe)?}}"
