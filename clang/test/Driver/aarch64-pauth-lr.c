// Check the -cc1 flags for the various forms of -mbranch-protection=pac-ret+pc.

// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc                  2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+b-key            2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-B-KEY
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+leaf             2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-LEAF
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+bti              2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-BTI
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+leaf+b-key+bti   2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-LEAF-B-KEY-BTI
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc                  -march=armv9.5-a 2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+b-key            -march=armv9.5-a 2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-B-KEY
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+leaf             -march=armv9.5-a 2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-LEAF
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+bti              -march=armv9.5-a 2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-BTI
// RUN: %clang --target=aarch64 -c %s -### -mbranch-protection=pac-ret+pc+leaf+b-key+bti   -march=armv9.5-a 2>&1 |  FileCheck %s --check-prefixes=PAUTH-LR-LEAF-B-KEY-BTI

// PAUTH-LR: "-msign-return-address=non-leaf" "-msign-return-address-key=a_key" "-mbranch-protection-pauth-lr"
// PAUTH-LR-B-KEY: "-msign-return-address=non-leaf" "-msign-return-address-key=b_key" "-mbranch-protection-pauth-lr"
// PAUTH-LR-LEAF: "-msign-return-address=all" "-msign-return-address-key=a_key" "-mbranch-protection-pauth-lr"
// PAUTH-LR-BTI: "-msign-return-address=non-leaf" "-msign-return-address-key=a_key" "-mbranch-protection-pauth-lr"
// PAUTH-LR-LEAF-B-KEY-BTI: "-msign-return-address=all" "-msign-return-address-key=b_key" "-mbranch-protection-pauth-lr" "-mbranch-target-enforce"

// NOT-PAUTH-LR: "-mbranch-target-enforce"
// NOT-PAUTH-LR-B-KEY: "-mbranch-target-enforce"
// NOT-PAUTH-LR-LEAF: "-mbranch-target-enforce"
// NOT-PAUTH-LR-BTI: "-mbranch-target-enforce"
