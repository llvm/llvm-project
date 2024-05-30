// RUN: %clang -arch armv6m -dM -E %s | FileCheck %s
// RUN: %clang -arch armv7m -dM -E %s | FileCheck %s
// RUN: %clang -arch armv7em -dM -E %s | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7m-apple-unknown-macho -dM -E %s | FileCheck %s

// CHECK-NOT: __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
// CHECK-NOT: __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
