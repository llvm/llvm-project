// RUN: %clang --target=aarch64-none-elf  --gcc-toolchain=%S/Inputs/basic_aarch64_gcc_tree -fno-integrated-as %s -### -c \
// RUN: 2>&1 | FileCheck %s

// CHECK: "{{.*}}as{{(.exe)?}}"
