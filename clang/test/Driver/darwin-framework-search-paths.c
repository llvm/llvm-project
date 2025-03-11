// UNSUPPORTED: system-windows
//   Windows is unsupported because we use the Unix path separator `/` in the test.

// Add default directories before running clang to check default
// search paths.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cp -R %S/Inputs/MacOSX15.1.sdk %t/
// RUN: mkdir -p %t/MacOSX15.1.sdk/System/Library/Frameworks
// RUN: mkdir -p %t/MacOSX15.1.sdk/System/Library/SubFrameworks
// RUN: mkdir -p %t/MacOSX15.1.sdk/usr/include

// RUN: %clang -xc %s -target arm64-apple-macosx15.1 -isysroot %t/MacOSX15.1.sdk -c -### 2>&1 \
// RUN: | FileCheck -DSDKROOT=%t/MacOSX15.1.sdk --check-prefix=CHECK-C %s
//
// CHECK-C:    "-isysroot" "[[SDKROOT]]"
// CHECK-C:    "-internal-externc-isystem" "[[SDKROOT]]/usr/include"
// CHECK-C:    "-iframework" "[[SDKROOT]]/System/Library/Frameworks"
// CHECK-C:    "-iframework" "[[SDKROOT]]/System/Library/SubFrameworks"

// RUN: %clang -xc++ %s -target arm64-apple-macosx15.1 -isysroot %t/MacOSX15.1.sdk -c -### 2>&1 \
// RUN: | FileCheck -DSDKROOT=%t/MacOSX15.1.sdk --check-prefix=CHECK-CXX %s
//
// CHECK-CXX:    "-isysroot" "[[SDKROOT]]"
// CHECK-CXX:    "-internal-externc-isystem" "[[SDKROOT]]/usr/include"
// CHECK-CXX:    "-iframework" "[[SDKROOT]]/System/Library/Frameworks"
// CHECK-CXX:    "-iframework" "[[SDKROOT]]/System/Library/SubFrameworks"
