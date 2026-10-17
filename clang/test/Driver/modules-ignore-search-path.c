// Check that -fmodules-ignore-search-path reaches cc1 through the driver.

// RUN: %clang -### -c -fmodules -fmodules-ignore-search-path=/tmp/gen1 \
// RUN:   -fmodules-ignore-search-path=/tmp/gen2 %s 2>&1 | FileCheck %s
//
// CHECK: "-fmodules-ignore-search-path=/tmp/gen1"
// CHECK-SAME: "-fmodules-ignore-search-path=/tmp/gen2"
