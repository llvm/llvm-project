// Check that we pass -fparse-all-comments to frontend.

// RUN: %clang -c %s -fparse-all-comments -### 2>&1 | FileCheck %s --check-prefix=CHECK-ARG

// CHECK-ARG: -fparse-all-comments
