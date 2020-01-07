// RUN: %clang -target x86_64-apple-ios13-macabi -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION1 %s

// CHECK-VERSION1: "x86_64-apple-ios13.0.0-macabi"

// FIXME: refuse versions earlier than ios13.
