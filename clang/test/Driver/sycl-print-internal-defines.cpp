// Test that clang can print defines in SYCL mode.

// RUN: %clangxx -fsycl -dM -E %s 2>&1 | FileCheck --check-prefix CHECK-PRINT-INTERNAL-DEFINES %s
// CHECK-PRINT-INTERNAL-DEFINES: #define

// Printing defines also works when input is stdin.
// RUN: %clangxx -fsycl -dM -E - < /dev/null 2>&1 | FileCheck --check-prefixes=CHECK-PRINT-INTERNAL-DEFINES,CHECK-NO-ERROR %s
// CHECK-NO-ERROR-NOT: error:
