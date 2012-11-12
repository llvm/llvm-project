// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1       \
// RUN:     -target i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/fake_install_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-4-7 %s

// CHECK-GCC-4-7: -fuse-init-array

// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1       \
// RUN:     -target i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-4-6 %s


// CHECK-GCC-4-6-NOT:  -fuse-init-array
