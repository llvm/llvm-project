// RUN: %clang -target x86_64-apple-darwin -arch arm64_32 -miphoneos-version-min=8.0 %s -### 2>&1 | FileCheck %s

// CHECK: "-cc1"{{.*}} "-triple" "arm64_32-apple-ios8.0.0"
// CHECK: ld{{.*}} "-arch" "arm64_32"
