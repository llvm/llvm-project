// RUN: %clang -### -target arm-none-none-eabi -march=armv8.5a+i8mm %s 2>&1 | FileCheck %s
// RUN: %clang -### -target aarch64-none-elf -march=armv8.5a+i8mm %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+i8mm"
// CHECK-NOT: "-target-feature" "-i8mm"

// RUN: %clang -### -target arm-none-none-eabi -march=armv8.6a+noi8mm %s 2>&1 | FileCheck %s --check-prefix=NOI8MM
// RUN: %clang -### -target aarch64-none-elf -march=armv8.6a+noi8mm %s 2>&1 | FileCheck %s --check-prefix=NOI8MM
// NOI8MM: "-target-feature" "-i8mm"
// NOI8MM-NOT: "-target-feature" "+i8mm"

// RUN: %clang -### -target arm-none-none-eabi %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: %clang -### -target aarch64-none-elf %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// ABSENT-NOT: "-target-feature" "+i8mm"
// ABSENT-NOT: "-target-feature" "-i8mm"
