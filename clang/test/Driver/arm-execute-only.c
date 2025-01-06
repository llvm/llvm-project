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
// RUN: echo "-DABC"  > %t.cfg
// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv8-m.main -mpure-code -mno-movt --config %t.cfg %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-PURE-CODE-NO-MOVT
// CHECK-PURE-CODE-NO-MOVT: error: option '-mpure-code' cannot be specified with '-mno-movt'

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv6-m -mexecute-only -fropi %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ROPI
// CHECK-NO-EXECUTE-ROPI: error: option '-mexecute-only' cannot be specified with '-fropi' for the thumbv6m sub-architecture

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv6-m -mexecute-only -frwpi %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-RWPI
// CHECK-NO-EXECUTE-RWPI: error: option '-mexecute-only' cannot be specified with '-frwpi' for the thumbv6m sub-architecture

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv6-m -mexecute-only -fpic %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-PIC
// CHECK-NO-EXECUTE-PIC: error: option '-mexecute-only' cannot be specified with '-fpic' for the thumbv6m sub-architecture

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv6-m -mexecute-only -fpie %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-PIE
// CHECK-NO-EXECUTE-PIE: error: option '-mexecute-only' cannot be specified with '-fpie' for the thumbv6m sub-architecture

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv6-m -mexecute-only -fPIC %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-PIC2
// CHECK-NO-EXECUTE-PIC2: error: option '-mexecute-only' cannot be specified with '-fPIC' for the thumbv6m sub-architecture

// RUN: not %clang -### --target=arm-arm-none-eabi -march=armv6-m -mexecute-only -fPIE %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-PIE2
// CHECK-NO-EXECUTE-PIE2: error: option '-mexecute-only' cannot be specified with '-fPIE' for the thumbv6m sub-architecture
