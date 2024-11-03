// RUN: %clang -fcolor-diagnostics -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-CD %s
// CHECK-CD: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fno-color-diagnostics -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-NCD %s
// CHECK-NCD-NOT: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fdiagnostics-color -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-DC %s
// CHECK-DC: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fno-diagnostics-color -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-NDC %s
// CHECK-NDC-NOT: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fdiagnostics-color=always -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-DCE_A %s
// CHECK-DCE_A: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fdiagnostics-color=never -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-DCE_N %s
// CHECK-DCE_N-NOT: "-cc1"{{.*}} "-fcolor-diagnostics"

// The test doesn't run in a PTY, so "auto" defaults to off.
// RUN: %clang -fdiagnostics-color=auto -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-DCE_AUTO %s
// CHECK-DCE_AUTO-NOT: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: not %clang -fdiagnostics-color=foo -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-DCE_FOO %s
// CHECK-DCE_FOO: error: invalid argument 'foo' to -fdiagnostics-color=

// Check that the last flag wins.
// RUN: %clang -fno-color-diagnostics -fdiagnostics-color -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-NCD_DC_S %s
// CHECK-NCD_DC_S: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fcolor-diagnostics -fno-diagnostics-color -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-CD_NDC_S %s
// CHECK-CD_NDC_S-NOT: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fdiagnostics-color -fno-color-diagnostics -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-DC_NCD_S %s
// CHECK-DC_NCD_S-NOT: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fno-diagnostics-color -fcolor-diagnostics -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-NDC_CD_S %s
// CHECK-NDC_CD_S: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fcolor-diagnostics -fdiagnostics-color=auto -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-CD_DCE_AUTO_S %s
// CHECK-CD_DCE_AUTO_S-NOT: "-cc1"{{.*}} "-fcolor-diagnostics"

// RUN: %clang -fansi-escape-codes -### -c %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CHECK-AEC %s
// CHECK-AEC: "-cc1"{{.*}} "-fansi-escape-codes"
