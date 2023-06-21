// RUN: %clang -### --target=aarch64-none-elf -march=armv8a+predres     %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+predres"
// CHECK-NOT: "-target-feature" "-predres"

// RUN: %clang -### --target=aarch64-none-elf -march=armv8.5a+nopredres %s 2>&1 | FileCheck %s --check-prefix=NOPR
// NOPR: "-target-feature" "-predres"
// NOPR-NOT: "-target-feature" "+predres"

// RUN: %clang -### --target=aarch64-none-elf                           %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// ABSENT-NOT: "-target-feature" "+predres"
// ABSENT-NOT: "-target-feature" "-predres"
