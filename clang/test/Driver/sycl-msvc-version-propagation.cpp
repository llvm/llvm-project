/// Verify that -fms-compatibility-version from the host toolchain reaches
/// the SPIR-V device cc1 when -fms-compatibility is implied by the aux-triple.

// RUN: %clang -### -fsycl --target=x86_64-pc-windows-msvc \
// RUN:   -x c++ %s 2>&1 | FileCheck %s

// CHECK: "-triple" "spirv64-unknown-unknown"
// CHECK-SAME: "-aux-triple" "x86_64-pc-windows-msvc
// CHECK-SAME: "-fms-compatibility"
// CHECK-SAME: "-fms-compatibility-version=19.
