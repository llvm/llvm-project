// Options for intel arch
// RUN: %clang -### -target x86_64-apple-darwin -fextend-arguments=32 %s 2>&1 \
// RUN: | FileCheck --implicit-check-not "-fextend-arguments=32"  %s
// RUN: %clang -### -target x86_64-apple-darwin -fextend-arguments=64 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-64 %s

// Unsupported target
// RUN: not %clang -target aarch64-unknown-windows-msvc -fextend-arguments=32 %s 2>&1 \
// RUN: | FileCheck -check-prefix=UNSUPPORTED-TARGET %s

// Invalid option value
// RUN: not %clang -target x86_64-apple-darwin -fextend-arguments=0 %s 2>&1 \
// RUN: | FileCheck -check-prefix=INVALID-VALUE %s

// CHECK-64: "-cc1" {{.*}}"-fextend-arguments=64"
// UNSUPPORTED-TARGET: error: unsupported option
// INVALID-VALUE: error: invalid argument '0' to -fextend-arguments
