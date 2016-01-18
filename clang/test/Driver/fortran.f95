// Check that the clang driver can invoke gcc to compile Fortran.

// RUN: %clang -target x86_64-unknown-linux-gnu -integrated-as -c %s -### 2>&1 \
// RUN:   | FileCheck %s
// CHECK: gcc
// CHECK: "-S"
// CHECK: "-x" "f95"
// CHECK: clang
// CHECK: "-cc1as"
