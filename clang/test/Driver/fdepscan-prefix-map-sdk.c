// RUN: %clang -fdepscan-prefix-map-sdk=/^sdk -### %s 2>&1 | FileCheck %s -check-prefix=NONE
// RUN: %clang -fdepscan-prefix-map-sdk=/^sdk -isysroot relative -### %s 2>&1 | FileCheck %s -check-prefix=NONE

// NONE-NOT: -fdepscan-prefix-map

// RUN: %clang -fdepscan-prefix-map-sdk=/^sdk -isysroot /sys/path -### %s 2>&1 | FileCheck %s
// RUN: %clang -fdepscan-prefix-map-sdk=/^sdk --sysroot /sys/path -### %s 2>&1 | FileCheck %s
// CHECK: -fdepscan-prefix-map=/sys/path=/^sdk
