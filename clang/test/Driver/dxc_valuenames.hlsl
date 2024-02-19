// RUN: %clang_dxc -### -T lib_6_6 -Fc - -fdiscard-value-names %s 2>&1 | FileCheck -check-prefix=CHECK-DISCARD-NAMES %s
// RUN: %clang_dxc -### -T lib_6_6 -Fc - -fno-discard-value-names %s 2>&1 | FileCheck -check-prefix=CHECK-NO-DISCARD-NAMES %s
// CHECK-DISCARD-NAMES: "-discard-value-names"
// CHECK-NO-DISCARD-NAMES-NOT: "-discard-value-names"
