// RUN: %clang -### -c --target=aarch64 -fno-plt -Werror %s 2>&1 | FileCheck %s --check-prefix=NOPLT
// RUN: %clang -### -c --target=x86_64 -fno-plt -Werror %s 2>&1 | FileCheck %s --check-prefix=NOPLT

// RUN: %clang -### -c --target=aarch64 -fno-plt -fplt -Werror %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c --target=powerpc64 -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=WARN,DEFAULT
// RUN: %clang -### -c --target=aarch64-windows -fno-plt %s 2>&1 | FileCheck %s --check-prefixes=WARN,DEFAULT

// WARN: warning: argument unused during compilation: '-fno-plt' [-Wunused-command-line-argument]
// NOPLT: "-fno-plt"
// DEFAULT-NOT: "-fno-plt"
