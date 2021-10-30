// RUN: %clang -target x86_64-apple-ios13.1-macabi -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION1 %s
// RUN: %clang -target x86_64-apple-ios-macabi -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION1 %s
// RUN: %clang -target x86_64-apple-ios13.0-macabi -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR %s
// RUN: %clang -target x86_64-apple-ios12.0-macabi -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR %s

// CHECK-VERSION1-NOT: error:
// CHECK-VERSION1: "x86_64-apple-ios13.1.0-macabi"
// CHECK-ERROR: error: invalid version number in '-target x86_64-apple-ios
