// RUN: %clang --target=aarch64-linux-pauthtest       --sysroot=%S/Inputs/multilib_aarch64_linux_tree -### -c %s 2>&1 | FileCheck %s
// RUN: %clang --target=aarch64-linux -mabi=pauthtest --sysroot=%S/Inputs/multilib_aarch64_linux_tree -### -c %s 2>&1 | FileCheck %s

// CHECK: "-internal-externc-isystem" "{{.*}}/usr/include/aarch64-linux-pauthtest"
