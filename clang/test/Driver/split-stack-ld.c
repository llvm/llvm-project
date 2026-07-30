// Test split stack ld flags.
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=i386-unknown-linux -fsplit-stack \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-I386 %s
//
// CHECK-LINUX-I386: "--wrap=pthread_create"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -fsplit-stack \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-X86-64 %s
//
// CHECK-LINUX-X86-64: "--wrap=pthread_create"
