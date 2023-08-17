// RUN: %clang --sysroot=%S/Inputs -c -fdriver-only -Werror --target=arm-arm-none-eabi \
// RUN:   -march=armv6-m -mexecute-only %s 2>&1 | count 0

// RUN: not %clang -### -c --target=arm-arm-none-eabi -march=armv6 -mexecute-only %s 2>&1 |    \
// RUN:   FileCheck --check-prefix CHECK-EXECUTE-ONLY-NOT-SUPPORTED %s
// CHECK-EXECUTE-ONLY-NOT-SUPPORTED: error: execute only is not supported for the armv6 sub-architecture

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv8-m.main -mexecute-only -mno-movt %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY-NO-MOVT
// CHECK-EXECUTE-ONLY-NO-MOVT: error: option '-mexecute-only' cannot be specified with '-mno-movt'

// RUN: %clang -### --target=arm-arm-none-eabi -march=armv7-m -x assembler -mexecute-only %s -c -### 2>&1 \
// RUN:    | FileCheck %s --check-prefix=CHECK-NO-EXECUTE-ONLY-ASM
// CHECK-NO-EXECUTE-ONLY-ASM: warning: argument unused during compilation: '-mexecute-only'

// -mpure-code flag for GCC compatibility
// RUN: not %clang -### -c --target=arm-arm-none-eabi -march=armv6 -mpure-code %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-EXECUTE-ONLY-NOT-SUPPORTED %s

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv8-m.main -mpure-code -mno-movt %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-PURE-CODE-NO-MOVT
// CHECK-PURE-CODE-NO-MOVT: error: option '-mpure-code' cannot be specified with '-mno-movt'
