// RUN: %clang -S -### -fopenacc %s 2>&1 | FileCheck %s --check-prefix=CHECK-DRIVER
// CHECK-DRIVER: "-cc1" {{.*}} "-fopenacc"
