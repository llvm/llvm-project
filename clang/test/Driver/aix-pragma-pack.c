// RUN: %clang -### -target powerpc-ibm-aix-xcoff -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### -target powerpc64-ibm-aix-xcoff -c %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: "-fxl-pragma-pack"
