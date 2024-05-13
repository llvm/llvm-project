// Verify -fdata-sections is the default for AIX

// RUN: %clang -### --target=powerpc-ibm-aix7.1.0.0 %s -c -o %t.o 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -### --target=powerpc64-ibm-aix7.1.0.0 %s -c -o %t.o 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-fdata-sections"
