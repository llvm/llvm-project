// RUN: %clang -### -target powerpc-ibm-aix-xcoff -msave-reg-params -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### -target powerpc64-ibm-aix-xcoff -msave-reg-params -c %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang -### -target powerpc-ibm-aix-xcoff -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLE
// RUN: %clang -### -target powerpc64-ibm-aix-xcoff -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLE

// CHECK: "-msave-reg-params"
// DISABLE-NOT: "-msave-reg-params"
