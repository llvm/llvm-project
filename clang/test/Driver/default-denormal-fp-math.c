// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s
// RUN: %clang -### -target i386-unknown-linux-gnu -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_linux_tree -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s

// RUN: %clang -### -target x86_64-scei-ps4 -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-PRESERVESIGN %s

// Flag omitted for default
// CHECK-IEEE-NOT: -fdenormal-fp-math
// CHECK-PRESERVESIGN: -fdenormal-fp-math=preserve-sign,preserve-sign
