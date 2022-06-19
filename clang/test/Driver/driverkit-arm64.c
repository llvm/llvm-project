// REQUIRES: aarch64-registered-target
// RUN: %clang %s -target arm64-apple-driverkit -### 2>&1 | FileCheck %s

// CHECK: "-target-cpu" "apple-a7"
