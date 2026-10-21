// REQUIRES: system-darwin

// Ensure that we never pick a version that's based on the SDK that's newer than
// the system version:
// RUN: rm -rf %t/SDKs/MacOSX99.99.99.sdk
// RUN: mkdir -p %t/SDKs/MacOSX99.99.99.sdk
// RUN: %clang -target x86_64-apple-darwin -isysroot %t/SDKs/MacOSX99.99.99.sdk %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MACOSX-SYSTEM-VERSION %s

// RUN: sed -e 's/15\.1/99\.99\.99/g' %S/Inputs/MacOSX15.1.sdk/SDKSettings.json > %t/SDKs/MacOSX99.99.99.sdk/SDKSettings.json
// RUN: %clang -target x86_64-apple-darwin -isysroot %t/SDKs/MacOSX99.99.99.sdk %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MACOSX-SYSTEM-VERSION %s

// CHECK-MACOSX-SYSTEM-VERSION-NOT: "-triple" "{{[[:alnum:]_-]*}}99.99.99"
