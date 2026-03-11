// REQUIRES: system-darwin

// Check that clang finds the macOS SDK automatically via DEFAULT_SYSROOT,
// even when SDKROOT is not set in the environment.
//
// RUN: env -u SDKROOT %clang -c %s -### 2>&1 | FileCheck %s
//
// CHECK: "-isysroot" "{{.*MacOSX[0-9\.]*\.sdk}}"
