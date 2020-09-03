// Check that the target cpu defaults to power4 on AIX.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -target powerpc-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT %s
// CHECK-MCPU-DEFAULT-NOT: warning:
// CHECK-MCPU-DEFAULT:     {{.*}}clang{{.*}}" "-cc1"
// CHECK-MCPU-DEFAULT:     "-target-cpu" "pwr4"

// Check that the user is able to overwrite the default with '-mcpu'.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -mcpu=pwr6 \
// RUN:        -target powerpc-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-USER %s
// CHECK-MCPU-USER-NOT: warning:
// CHECK-MCPU-USER:     {{.*}}clang{{.*}}" "-cc1"
// CHECK-MCPU-USER:     "-target-cpu" "pwr6"
