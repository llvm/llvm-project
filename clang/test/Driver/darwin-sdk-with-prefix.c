// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir

// RUN: rm -rf %t.dir/prefix.iPhoneOS12.0.0.sdk
// RUN: mkdir -p %t.dir/prefix.iPhoneOS12.0.0.sdk
// RUN: %clang -c -isysroot %t.dir/prefix.iPhoneOS12.0.0.sdk -target arm64-apple-darwin %s -### 2>&1 | FileCheck %s
// RUN: env SDKROOT=%t.dir/prefix.iPhoneOS12.0.0.sdk %clang -c -target arm64-apple-darwin %s -### 2>&1 | FileCheck %s
//
// CHECK-NOT: warning: using sysroot for
// CHECK: "-triple" "arm64-apple-ios12.0.0"
