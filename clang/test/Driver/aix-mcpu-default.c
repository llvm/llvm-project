// Check that the target cpu defaults to power7 on AIX7.2 and up.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -target powerpc-ibm-aix7.2 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 on AIX7.2 and up.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -target powerpc64-ibm-aix7.2 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 on AIX7.1 and below.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -target powerpc-ibm-aix7.1 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 on AIX7.1 and below.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -target powerpc64-ibm-aix7.1 \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 when level not specified.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -target powerpc-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// Check that the target cpu defaults to power7 when level not specified.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -target powerpc64-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-DEFAULT-AIX72 %s

// CHECK-MCPU-DEFAULT-AIX72-NOT: warning:
// CHECK-MCPU-DEFAULT-AIX72:     {{.*}}clang{{.*}}" "-cc1"
// CHECK-MCPU-DEFAULT-AIX72:     "-target-cpu" "pwr7"

// Check that the user is able to overwrite the default with '-mcpu'.
// RUN: %clang -no-canonical-prefixes %s -### -c 2>&1 \
// RUN:        -mcpu=pwr6 \
// RUN:        -target powerpc-ibm-aix \
// RUN:   | FileCheck --check-prefix=CHECK-MCPU-USER %s
// CHECK-MCPU-USER-NOT: warning:
// CHECK-MCPU-USER:     {{.*}}clang{{.*}}" "-cc1"
// CHECK-MCPU-USER:     "-target-cpu" "pwr6"
