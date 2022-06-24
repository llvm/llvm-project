// RUN: not %clang -target arm-unknown-linux -marm -mcpu=cortex-m0 %s -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix M0 %s
// M0: error: CPU 'cortex-m0' does not support 'ARM' execution mode

// RUN: not %clang -target arm-unknown-linux -marm -march=armv7m %s -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix ARMV7M %s
// RUN: not %clang -target armv7m-unknown-linux -mno-thumb %s -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix ARMV7M %s
// ARMV7M: error: architecture 'armv7m' does not support 'ARM' execution mode

// RUN: %clang -S -emit-llvm -target arm-unknown-linux -mcpu=cortex-m0 %s -o /dev/null 2>&1
// M-Profile CPUs default to Thumb mode even if arm triples are provided.
