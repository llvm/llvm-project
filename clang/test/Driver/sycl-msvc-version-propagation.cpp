/// Test that the host -fms-compatibility-version reaches the SPIR-V device cc1.

// RUN: %clang -### -fsycl --target=x86_64-pc-windows-msvc \
// RUN:   -x c++ %s 2>&1 | FileCheck %s

// CHECK: "-triple" "spirv64-unknown-unknown"
// CHECK-SAME: "-aux-triple" "x86_64-pc-windows-msvc
// CHECK-SAME: "-fms-compatibility"
// CHECK-SAME: "-fms-compatibility-version=19.

/// With no detectable host version, the driver falls back to the lowest
/// supported one.

// RUN: %clang_cl -### -fsycl -fno-ms-extensions --target=x86_64-pc-windows-msvc \
// RUN:   /winsysroot %t.emptyroot -- %s 2>&1 | FileCheck --check-prefix=FALLBACK %s

// FALLBACK: "-triple" "spirv64-unknown-unknown"
// FALLBACK-SAME: "-aux-triple" "x86_64-pc-windows-msvc
// FALLBACK-SAME: "-fms-compatibility-version=19.16.27023"
