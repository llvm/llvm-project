// REQUIRES: system-darwin

// Check that we default to running `xcrun --show-sdk-path` if there is no
// SDKROOT defined in the environment.
//
// RUN: env -u SDKROOT %clang -target x86_64-apple-darwin -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-XC < %t.log %s
//
// CHECK-XC: clang
// CHECK-XC: "-cc1"
// CHECK-XC: "-isysroot" "{{.*MacOSX[0-9\.]*\.sdk}}"

// Check once again that we default to running `xcrun`, this time with another target.
//
// RUN: env -u SDKROOT %clang -target arm64-apple-ios -c %s -### 2> %t.log
// RUN: FileCheck --check-prefix=CHECK-XC-IOS < %t.log %s
//
// CHECK-XC-IOS: clang
// CHECK-XC-IOS: "-cc1"
// CHECK-XC-IOS: "-isysroot" "{{.*iPhoneOS[0-9\.]*\.sdk}}"
