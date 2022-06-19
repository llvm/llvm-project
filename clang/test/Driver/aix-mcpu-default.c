// Check that the target cpu defaults to power7 on AIX7.2 and up.
// RUN: %clang %s -### -c 2>&1 --target=powerpc-ibm-aix7.2 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 on AIX7.2 and up.
// RUN: %clang %s -### -c 2>&1 --target=powerpc64-ibm-aix7.2 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 on AIX7.1 and below.
// RUN: %clang %s -### -c 2>&1 --target=powerpc-ibm-aix7.1 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 on AIX7.1 and below.
// RUN: %clang %s -### -c 2>&1 --target=powerpc64-ibm-aix7.1 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 when level not specified.
// RUN: %clang %s -### -c 2>&1 --target=powerpc-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 when level not specified.
// RUN: %clang %s -### -c 2>&1 --target=powerpc64-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// CHECK-MCPU-DEFAULT-AIX72-NOT: warning:
// CHECK-MCPU-DEFAULT-AIX72:     "-cc1"
// CHECK-MCPU-DEFAULT-AIX72:     "-target-cpu" "pwr7"

// Check that the user is able to overwrite the default with '-mcpu'.
// RUN: %clang %s -### -c 2>&1 -mcpu=pwr6 --target=powerpc-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-USER %s
// CHECK-MCPU-USER-NOT: warning:
// CHECK-MCPU-USER:     "-cc1"
// CHECK-MCPU-USER:     "-target-cpu" "pwr6"
