// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mcmodel=medium -mtocdata %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mcmodel=large -mtocdata %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mtocdata %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mcmodel=medium -mtocdata %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mcmodel=large -mtocdata %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mtocdata %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-cc1"{{.*}}" "-mtocdata"
