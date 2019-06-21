// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-tapi-elf-v1 %s | \
// RUN: FileCheck -check-prefix=CHECK-TAPI %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -interface-stub-version=experimental-tapi-elf-v1 %s | \
// RUN: FileCheck -check-prefix=CHECK-SYMBOLS %s
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SYMBOLS %s

// CHECK-TAPI: data: { Type: Object, Size: 4 }
// CHECK-SYMBOLS: data
int data = 42;
