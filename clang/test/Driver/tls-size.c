// Options for AArch64 ELF
// RUN: %clang -### -target aarch64-linux-gnu -mtls-size=12 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-12 %s
// RUN: %clang -### -target aarch64-linux-gnu -mtls-size=24 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-24 %s
// RUN: %clang -### -target aarch64-linux-gnu -mtls-size=32 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-32 %s
// RUN: %clang -### -target aarch64-linux-gnu -mtls-size=48 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-48 %s

// Unsupported target
// RUN: not %clang -target aarch64-unknown-windows-msvc -mtls-size=24 %s 2>&1 \
// RUN: | FileCheck -check-prefix=UNSUPPORTED-TARGET %s
// RUN: not %clang -target x86_64-linux-gnu -mtls-size=24 %s 2>&1 \
// RUN: | FileCheck -check-prefix=UNSUPPORTED-TARGET %s

// Invalid option value
// RUN: not %clang -target aarch64-linux-gnu -mtls-size=0 %s 2>&1 \
// RUN: | FileCheck -check-prefix=INVALID-VALUE %s

// CHECK-12: "-cc1" {{.*}}"-mtls-size=12"
// CHECK-24: "-cc1" {{.*}}"-mtls-size=24"
// CHECK-32: "-cc1" {{.*}}"-mtls-size=32"
// CHECK-48: "-cc1" {{.*}}"-mtls-size=48"
// UNSUPPORTED-TARGET: error: unsupported option
// INVALID-VALUE: error: invalid integral value
