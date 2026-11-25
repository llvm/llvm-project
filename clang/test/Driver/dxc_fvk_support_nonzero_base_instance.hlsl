// Verify that -fvk-support-nonzero-base-instance is accepted by the driver and
// forwarded to cc1 as -fvk-support-nonzero-base-instance.
// RUN: %clang_dxc -spirv -Tvs_6_7 -fvk-support-nonzero-base-instance -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-FLAG
// CHECK-FLAG: "-fvk-support-nonzero-base-instance"

// Without the flag, it must not appear in cc1 args.
// RUN: %clang_dxc -spirv -Tvs_6_7 -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-FLAG
// CHECK-NO-FLAG-NOT: "-fvk-support-nonzero-base-instance"

// The flag requires -spirv.
// RUN: not %clang_dxc -Tvs_6_7 -fvk-support-nonzero-base-instance -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-SPIRV
// CHECK-NO-SPIRV: error: invalid argument '-fvk-support-nonzero-base-instance' only allowed with '-spirv'
