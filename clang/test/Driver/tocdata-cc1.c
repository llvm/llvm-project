// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mcmodel=medium -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOTOC %s
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mcmodel=large -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOTOC %s
// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mcmodel=medium -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOTOC %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mcmodel=large -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NOTOC %s
// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mtocdata %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-TOC %s
// CHECK-NOTOC: warning: ignoring '-mtocdata' as it is only supported for -mcmodel=small
// CHECK-NOTOC-NOT: "-cc1"{{.*}}" "-mtocdata"
// CHECK-TOC: "-cc1"{{.*}}" "-mtocdata"
// CHECK-TOC-NOT: warning: ignoring '-mtocdata' as it is only supported for -mcmodel=small
