// RUN: %clang -target arm64-apple-darwin27 -fuse-ld= \
// RUN:   -mlinker-version=520 %s -### 2>&1 | FileCheck %s

// CHECK: "-cc1" "-triple" "arm64-apple-macosx27.0.0"
// CHECK: "-platform_version" "macos" "27.0.0"
