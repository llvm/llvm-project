// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s
// RUN: %clang -### -target i386-unknown-linux-gnu -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s

// RUN: %clang -### -target x86_64-unknown-linux-gnu --sysroot=%S/Inputs/basic_linux_tree -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s

// crtfastmath enables ftz and daz
// RUN: %clang -### -target x86_64-unknown-linux-gnu -ffast-math --sysroot=%S/Inputs/basic_linux_tree -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-PRESERVESIGN %s

// crt not linked in with nostartfiles
// RUN: %clang -### -target x86_64-unknown-linux-gnu -ffast-math -nostartfiles --sysroot=%S/Inputs/basic_linux_tree -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s

// If there's no crtfastmath, don't assume ftz/daz
// RUN: %clang -### -target x86_64-unknown-linux-gnu -ffast-math --sysroot=/dev/null -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s

// RUN: %clang -### -target x86_64-scei-ps4 -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-PRESERVESIGN %s

// Flag omitted for default
// CHECK-IEEE-NOT: -fdenormal-fp-math
// CHECK-PRESERVESIGN: -fdenormal-fp-math=preserve-sign,preserve-sign
