// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mcmodel=medium -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mcmodel=large -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mcmodel=medium -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mcmodel=large -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// CHECK-TOC: "-cc1"{{.*}}" "-mtocdata"
