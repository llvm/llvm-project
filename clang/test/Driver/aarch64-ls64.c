// Test that target feature ls64 is implemented and available correctly

// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.7-a+ls64 %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+ls64"

// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.7-a+nols64 %s 2>&1 | FileCheck %s --check-prefix=NO_LS64
// NO_LS64: "-target-feature" "-ls64"

// The LD64B/ST64B accelerator extension is disabled by default.
// RUN: %clang -### --target=aarch64-none-none-eabi                  %s 2>&1 | FileCheck %s --check-prefix=ABSENT_LS64
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.7-a %s 2>&1 | FileCheck %s --check-prefix=ABSENT_LS64
// RUN: %clang -### --target=aarch64-none-none-eabi -march=armv8.7-a %s 2>&1 | FileCheck %s --check-prefix=ABSENT_LS64
// ABSENT_LS64-NOT: "-target-feature" "+ls64"
// ABSENT_LS64-NOT: "-target-feature" "-ls64"
