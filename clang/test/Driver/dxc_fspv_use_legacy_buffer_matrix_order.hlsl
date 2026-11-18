// Verify that -fspv-use-legacy-buffer-matrix-order is accepted by the driver
// and forwarded to cc1 as -fspv-use-legacy-buffer-matrix-order.
// RUN: %clang_dxc -spirv -Tlib_6_7 -fspv-use-legacy-buffer-matrix-order -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-LEGACY
// CHECK-LEGACY: "-fspv-use-legacy-buffer-matrix-order"

// Without the flag, -fspv-use-legacy-buffer-matrix-order must not appear in
// cc1 args.
// RUN: %clang_dxc -spirv -Tlib_6_7 -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-LEGACY
// CHECK-NO-LEGACY-NOT: "-fspv-use-legacy-buffer-matrix-order"

// The flag requires -spirv.
// RUN: not %clang_dxc -Tlib_6_7 -fspv-use-legacy-buffer-matrix-order -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-SPIRV
// CHECK-NO-SPIRV: error: invalid argument '-fspv-use-legacy-buffer-matrix-order' only allowed with '-spirv'

