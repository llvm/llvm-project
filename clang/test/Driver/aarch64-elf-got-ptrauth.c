// REQUIRES: aarch64-registered-target

// Test default behavior of ptrauth-elf-got flag: if the user passes neither
// the enable nor disable flag, the driver automatically enables it.

// RUN: %clang -### -c --target=aarch64-linux-pauthtest %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// DEFAULT: "-cc1"
// DEFAULT: "-fptrauth-elf-got"

// Explicitly enabled.
// RUN: %clang -### -c --target=aarch64-linux-pauthtest -fptrauth-elf-got %s 2>&1 | FileCheck %s --check-prefix=ENABLE
// ENABLE: "-cc1"
// ENABLE: "-fptrauth-elf-got"

// Explicitly disabled.
// RUN: %clang -### -c --target=aarch64-linux-pauthtest -fno-ptrauth-elf-got %s 2>&1 | FileCheck %s --check-prefix=DISABLE
// DISABLE: "-cc1"
// DISABLE-NOT: "-fptrauth-elf-got"
