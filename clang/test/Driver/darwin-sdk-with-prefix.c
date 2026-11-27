// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir

// The name of the SDK directory doesn't matter, the supported triples come from the SDKSettings file.

// RUN: rm -rf %t.dir/prefix.MacOSX13.0.sdk
// RUN: cp -R %S/Inputs/iPhoneOS13.0.sdk %t.dir/prefix.MacOSX13.0.sdk
// RUN: %clang -c -isysroot %t.dir/prefix.MacOSX13.0.sdk -target arm64-apple-darwin %s -### 2>&1 | FileCheck %s
// RUN: env SDKROOT=%t.dir/prefix.MacOSX13.0.sdk %clang -c -target arm64-apple-darwin %s -### 2>&1 | FileCheck %s
//
// CHECK-NOT: warning: using sysroot for
// CHECK: "-triple" "arm64-apple-ios13.0.0"

// RUN: %clang -c -isysroot %t.dir/prefix.MacOSX13.0.sdk -target arm64-apple-macos %s -### 2>&1 | FileCheck %s --check-prefix=INCOMPATIBLE
// RUN: env SDKROOT=%t.dir/prefix.MacOSX13.0.sdk %clang -c -target arm64-apple-macos %s -### 2>&1 | FileCheck %s --check-prefix=INCOMPATIBLE
//
// INCOMPATIBLE: warning: using sysroot for
// INCOMPATIBLE: "-triple" "arm64-apple-macos{{[\d.]*}}
