// Check that we pass -fretain-comments-from-system-headers to frontend.
//
// RUN: %clang -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-RETAIN
// RUN: %clang -c %s -fretain-comments-from-system-headers -### 2>&1 | FileCheck %s --check-prefix=CHECK-RETAIN
//
// CHECK-RETAIN: -fretain-comments-from-system-headers
//
// CHECK-NO-RETAIN-NOT: -fretain-comments-from-system-headers

