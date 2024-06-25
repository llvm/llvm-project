// RUN: %clang -### --target=x86_64 -fbasic-block-address-map %s -S 2>&1 | FileCheck -check-prefix=CHECK-PRESENT %s
// RUN: %clang -### --target=aarch64 -fbasic-block-address-map %s -S 2>&1 | FileCheck -check-prefix=CHECK-PRESENT %s
// CHECK-PRESENT: -fbasic-block-address-map

// RUN: %clang -### --target=x86_64 -fno-basic-block-address-map %s -S 2>&1 | FileCheck %s --check-prefix=CHECK-ABSENT
// CHECK-ABSENT-NOT: -fbasic-block-address-map

// RUN: not %clang -c --target=x86_64-apple-darwin10 -fbasic-block-address-map %s -S 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s
// CHECK-TRIPLE: error: unsupported option '-fbasic-block-address-map' for target
