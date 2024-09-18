// Test that target feature ls64 is implemented and available correctly

// RUN: %clang -### --target=aarch64-none-elf -march=armv8.7-a+ls64 %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+ls64"
// CHECK: "-target-feature" "+ls64_accdata"
// CHECK: "-target-feature" "+ls64_v"

// RUN: %clang -### --target=aarch64-none-elf -march=armv8.7-a+nols64 %s 2>&1 | FileCheck %s --check-prefix=ABSENT_LS64

// The LD64B/ST64B accelerator extension is disabled by default.
// RUN: %clang -### --target=aarch64-none-elf                  %s 2>&1 | FileCheck %s --check-prefix=ABSENT_LS64
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.7-a %s 2>&1 | FileCheck %s --check-prefix=ABSENT_LS64
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.7-a %s 2>&1 | FileCheck %s --check-prefix=ABSENT_LS64
// ABSENT_LS64-NOT: "-target-feature" "+ls64"
// ABSENT_LS64-NOT: "-target-feature" "-ls64"

// Test that nols64 disables all three FEAT_LS64, FEAT_LS64_V and FEAT_LS64_ACCDATA.
// RUN: %clang -### --target=aarch64-none-elf -march=armv8.7-a+ls64+nols64 %s 2>&1 | FileCheck %s --check-prefix=NOLS64
// NOLS64: "-target-feature" "-ls64"
// NOLS64: "-target-feature" "-ls64_accdata"
// NOLS64: "-target-feature" "-ls64_v"
