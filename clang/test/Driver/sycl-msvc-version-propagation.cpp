/// Test that -fms-compatibility-version is consistent between the host and the
/// SPIR-V device compilation.

// RUN: %clang -### -fsycl --target=x86_64-pc-windows-msvc \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl \
// RUN:   -x c++ %s 2>&1 | FileCheck %s

// CHECK: "-triple" "spirv64-unknown-unknown"
// CHECK-SAME: "-aux-triple" "x86_64-pc-windows-msvc
// CHECK-SAME: "-fms-compatibility"
// CHECK-SAME: "-fms-compatibility-version=[[MSVC_VER:[0-9.]+]]"

// CHECK: "-triple" "x86_64-pc-windows-msvc
// CHECK-SAME: "-fms-compatibility"
// CHECK-SAME: "-fms-compatibility-version=[[MSVC_VER]]"
