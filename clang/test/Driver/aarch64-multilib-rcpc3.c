// RUN: %clang --target=aarch64-none-elf -march=armv8.9-a+rcpc3 -print-multi-flags-experimental -c %s 2>&1 | FileCheck %s

// CHECK: -march=armv8.9-a
// CHECK-SAME: +rcpc+rcpc3+
