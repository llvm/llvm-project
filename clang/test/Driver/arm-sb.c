// RUN: %clang -### -target arm-none-none-eabi -march=armv8a+sb %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-elf -march=armv8a+sb %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-elf -mcpu=cortex-a510 %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+sb"
// CHECK-NOT: "-target-feature" "-sb"

// RUN: %clang -### -target arm-none-none-eabi -march=armv8.5a+nosb %s 2>&1 | FileCheck %s --check-prefix=NOSB
// RUN: %clang -### -target aarch64-none-elf -mcpu=cortex-a510+nosb %s 2>&1 | FileCheck %s --check-prefix=NOSB
// NOSB: "-target-feature" "-sb"
// NOSB-NOT: "-target-feature" "+sb"

// RUN: %clang -### -target arm-none-none-eabi %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: %clang -### -target aarch64-none-elf %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: %clang -### -target aarch64-none-elf -march=armv8.5a+nosb %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// ABSENT-NOT: "-target-feature" "+sb"
// ABSENT-NOT: "-target-feature" "-sb"
