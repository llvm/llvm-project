// RUN: %clang -### --target=aarch64-none-elf -march=armv8.4a+pmuv3 %s 2>&1 | FileCheck --check-prefix=CHECK-PMUV3 %s
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.2a+pmuv3 %s 2>&1 | FileCheck --check-prefix=CHECK-PMUV3 %s
// CHECK-PMUV3: "-target-feature" "+pmuv3"

// RUN: %clang -### --target=aarch64-none-elf -mcpu=cortex-a520+nopmuv3 %s 2>&1 | FileCheck --check-prefix=CHECK-NO-PMUv3 %s
// CHECK-NO-PMUv3: "-target-feature" "-pmuv3"

// RUN: %clang -### --target=aarch64-none-elf                 %s 2>&1 | FileCheck %s --check-prefix=ABSENT-PMUV3
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.4a %s 2>&1 | FileCheck %s --check-prefix=ABSENT-PMUV3
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.2a %s 2>&1 | FileCheck %s --check-prefix=ABSENT-PMUV3
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.4a+nopmuv3 %s 2>&1 | FileCheck --check-prefix=ABSENT-PMUV3 %s
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.2a+nopmuv3 %s 2>&1 | FileCheck --check-prefix=ABSENT-PMUV3 %s
// ABSENT-PMUV3-NOT: "-target-feature" "+pmuv3"
// ABSENT-PMUV3-NOT: "-target-feature" "-pmuv3"
