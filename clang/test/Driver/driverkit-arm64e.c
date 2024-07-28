// RUN: %clang %s -target arm64e-apple-driverkit -### 2>&1 | FileCheck %s

// CHECK: "-target-cpu" "apple-a12"
