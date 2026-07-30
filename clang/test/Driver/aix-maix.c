// Check GCC AIX bitmode compat options.

// RUN: %clang -target powerpc-ibm-aix -maix64 -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK64 %s

// RUN: %clang -target powerpc64-ibm-aix -maix32 -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK32 %s

// RUN: not %clang --target=powerpc-unknown-linux -maix64 -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=ERROR %s

// RUN: not %clang --target=powerpc64-unknown-linux -maix32 -### -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=ERROR %s

// CHECK32: Target: powerpc-ibm-aix
// CHECK64: Target: powerpc64-ibm-aix
// ERROR: error: unsupported option '-maix
