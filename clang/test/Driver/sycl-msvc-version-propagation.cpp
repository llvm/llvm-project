/// Verify that -fms-compatibility-version from the host toolchain reaches
/// the SPIR-V device cc1 when -fms-compatibility is implied by the aux-triple.

// RUN: %clang -### -fsycl --target=x86_64-pc-windows-msvc \
// RUN:   -x c++ %s 2>&1 | FileCheck %s

// CHECK: "-triple" "spirv64-unknown-unknown"
// CHECK-SAME: "-aux-triple" "x86_64-pc-windows-msvc
// CHECK-SAME: "-fms-compatibility"
// CHECK-SAME: "-fms-compatibility-version=19.

/// When the host toolchain cannot determine a version either (e.g. with
/// -fno-ms-extensions), the driver falls back to the lowest supported MSVC
/// version so device code still gets a consistent -fms-compatibility-version.

// RUN: %clang -### -fsycl -fno-ms-extensions --target=x86_64-pc-windows-msvc \
// RUN:   -x c++ %s 2>&1 | FileCheck --check-prefix=FALLBACK %s

// FALLBACK: "-triple" "spirv64-unknown-unknown"
// FALLBACK-SAME: "-aux-triple" "x86_64-pc-windows-msvc
// FALLBACK-SAME: "-fms-compatibility-version=19.16.27023"
