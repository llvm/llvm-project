// RUN: %clang --sysroot=%S/Inputs -c -fdriver-only -Werror --target=aarch64-unknown-linux-gnu \
// RUN:   -mexecute-only %s 2>&1 | count 0

// RUN: %clang -### --target=aarch64-unknown-linux-gnu -x assembler -mexecute-only %s -c -### 2>&1 \
// RUN:    | FileCheck %s --check-prefix=CHECK-NO-EXECUTE-ONLY-ASM
// CHECK-NO-EXECUTE-ONLY-ASM: warning: argument unused during compilation: '-mexecute-only'
