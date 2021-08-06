// Check that setting an OBJECT_MODE converts the AIX triple to the right variant.
// RUN: env OBJECT_MODE=64 \
// RUN: %clang -target powerpc-ibm-aix -print-target-triple | FileCheck -check-prefix=CHECK64 %s

// RUN: env OBJECT_MODE=32 \
// RUN: %clang -target powerpc64-ibm-aix -print-target-triple | FileCheck -check-prefix=CHECK32 %s

// Command-line options win.
// RUN: env OBJECT_MODE=64 \
// RUN: %clang -target powerpc64-ibm-aix -print-target-triple -m32 | FileCheck -check-prefix=CHECK32 %s

// RUN: env OBJECT_MODE=32 \
// RUN: %clang -target powerpc-ibm-aix -print-target-triple -m64 | FileCheck -check-prefix=CHECK64 %s

// CHECK32: powerpc-ibm-aix
// CHECK64: powerpc64-ibm-aix

// Emit a diagnostic if there is an invalid mode.
// RUN: env OBJECT_MODE=31 \
// RUN: not %clang -target powerpc-ibm-aix 2>&1 | FileCheck -check-prefix=DIAG %s

// DIAG: error: OBJECT_MODE setting 31 is not recognized and is not a valid setting
