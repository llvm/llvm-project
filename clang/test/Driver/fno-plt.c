// RUN: %clang -### -c --target=aarch64 -fno-plt -Werror %s 2>&1 | FileCheck %s
// RUN: %clang -### -c --target=aarch64 -fno-plt -fplt -Werror %s 2>&1 | FileCheck %s --check-prefix=NO
// RUN: %clang -### -c --target=aarch64-windows -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=WARN,NO

// WARN: warning: argument unused during compilation: '-fno-plt' [-Wunused-command-line-argument]
// CHECK: "-fno-plt"
// NO-NOT: "-fno-plt"
