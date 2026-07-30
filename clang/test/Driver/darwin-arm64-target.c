// RUN: %clang -target aarch64-apple-darwin %s -miphoneos-version-min=8.0 -### 2>&1 | FileCheck %s

// CHECK: "-cc1"{{.*}} "-triple" "arm64-apple-ios8.0.0"
