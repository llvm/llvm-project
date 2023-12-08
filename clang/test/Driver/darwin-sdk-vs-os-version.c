// REQUIRES: system-darwin

// Ensure that we never pick a version that's based on the SDK that's newer than
// the system version:
// RUN: rm -rf %t/SDKs/MacOSX99.99.99.sdk
// RUN: mkdir -p %t/SDKs/MacOSX99.99.99.sdk
// RUN: %clang -target x86_64-apple-darwin -isysroot %t/SDKs/MacOSX99.99.99.sdk %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MACOSX-SYSTEM-VERSION %s

// CHECK-MACOSX-SYSTEM-VERSION-NOT: 99.99.99"
