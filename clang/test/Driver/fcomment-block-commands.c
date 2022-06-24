// Check that we pass -fcomment-block-commands to frontend.

// RUN: %clang -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ARG
// RUN: %clang -c %s -fcomment-block-commands=Foo -### 2>&1 | FileCheck %s --check-prefix=CHECK-ARG

// CHECK-ARG: -fcomment-block-commands=Foo

// CHECK-NO-ARG-NOT: -fcomment-block-commands=
