// RUN: not %clang --target=s390x -c -### %s -mpacked-stack -mbackchain 2>&1 | FileCheck %s
// RUN: %clang -target s390x -c -### %s -mpacked-stack -mbackchain -msoft-float \
// RUN:   2>&1 | FileCheck %s --check-prefix=KERNEL-BUILD
// REQUIRES: systemz-registered-target

// CHECK: error: unsupported option '-mpacked-stack -mbackchain -mhard-float'
// KERNEL-BUILD-NOT: error: unsupported option
