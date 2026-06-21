// UNSUPPORTED: system-windows
// Test that clang can print defines in SYCL mode.
// REQUIRES: x86-registered-target

// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -dM -E %s 2>&1 \
// RUN: | FileCheck --check-prefix CHECK-PRINT-INTERNAL-DEFINES %s
// CHECK-PRINT-INTERNAL-DEFINES: #define

// Printing defines also works when input is stdin.
// RUN: %clangxx -target x86_64-unknown-linux-gnu -fsycl -dM -E - < /dev/null 2>&1 \
// RUN: | FileCheck --check-prefixes=CHECK-PRINT-INTERNAL-DEFINES,CHECK-NO-ERROR %s
// CHECK-NO-ERROR-NOT: error:
