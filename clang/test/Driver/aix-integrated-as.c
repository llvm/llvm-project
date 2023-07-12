// Test integrated-as is called by default on AIX.

// Check powerpc-ibm-aix7.1.0.0, 32-bit.
// RUN: %clang %s -### -c 2>&1 \
// RUN:         --target=powerpc-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-IAS32 %s
// CHECK-IAS32-NOT: "-a32"
// CHECK-IAS32: "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"{{.*}}"-emit-obj"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit.
// RUN: %clang %s -### -c 2>&1 \
// RUN:         --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:   | FileCheck --check-prefix=CHECK-IAS64 %s
// CHECK-IAS64-NOT: "-a64"
// CHECK-IAS64: "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"{{.*}}"-emit-obj"
