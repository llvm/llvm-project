// REQUIRES: system-darwin

// RUN: env XROS_DEPLOYMENT_TARGET=1.0 %clang -arch arm64 -c -### %s 2>&1 | FileCheck %s

// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/XROS1.0.sdk
// RUN: %clang -arch arm64 -isysroot %t.dir/XROS1.0.sdk -c -### %s 2>&1 | FileCheck %s
// RUN: mkdir -p %t.dir/XRSimulator1.0.sdk
// RUN: %clang -arch arm64 -isysroot %t.dir/XRSimulator1.0.sdk -c -### %s 2>&1 | FileCheck --check-prefix=CHECK_SIM %s


// CHECK: "-cc1"{{.*}} "-triple" "arm64-apple-xros1.0.0"
// CHECK_SIM: "-cc1"{{.*}} "-triple" "arm64-apple-xros1.0.0-simulator"
