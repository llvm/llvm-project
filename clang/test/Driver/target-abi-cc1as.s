// Check if -cc1as knows about the 'target-abi' argument.
// REQUIRES: mips-registered-target

// RUN: %clang -cc1as -triple mips--linux-gnu -filetype obj -target-cpu mips32 -target-abi o32 %s 2>&1 | \
// RUN:   FileCheck %s
// CHECK-NOT: clang -cc1as: error: unknown argument: '-target-abi'
