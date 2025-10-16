// REQUIRES: systemz-registered-target

// RUN: %clang -### -target s390x-ibm-zos -c %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: "-fxl-pragma-pack"

// RUN: %clang -### -fno-xl-pragma-pack -target s390x-ibm-zos -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=NOOPT
// NOOPT-NOT: "-fxl-pragma-pack"

