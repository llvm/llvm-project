// Verify that -fspv-preserve-interface is accepted by the driver and forwarded
// to cc1 as -fspv-preserve-interface.
// RUN: %clang_dxc -spirv -Tlib_6_7 -fspv-preserve-interface -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PRESERVE
// CHECK-PRESERVE: "-fspv-preserve-interface"

// Without the flag, -fspv-preserve-interface must not appear in cc1 args.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PRESERVE
// CHECK-NO-PRESERVE-NOT: "-fspv-preserve-interface"
