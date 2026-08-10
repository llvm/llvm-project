// RUN: not %clang -target x86_64-apple-ios13.0-macabi -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR %s
// RUN: not %clang -target x86_64-apple-ios12.0-macabi -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-ERROR %s

// CHECK-ERROR: error: invalid version number in '-target x86_64-apple-ios
