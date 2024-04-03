// RUN: %clang --target=armv6m-apple-darwin -dM -E %s | FileCheck %s
// RUN: %clang --target=armv7m-apple-darwin -dM -E %s | FileCheck %s
// RUN: %clang --target=armv7em-apple-darwin -dM -E %s | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7m-apple-unknown-macho -dM -E %s | FileCheck %s

// CHECK-NOT: __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
// CHECK-NOT: __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
