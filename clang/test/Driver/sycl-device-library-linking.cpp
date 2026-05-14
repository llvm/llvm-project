// Test that SYCL device libraries are linked at compile-time for SPIR/SPIRV targets

// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-DEVICE-LIBS

// RUN: %clangxx -fsycl --no-offloadlib %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-NO-DEVICE-LIBS

// CHECK-DEVICE-LIBS: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-DEVICE-LIBS-SAME: "-fsycl-is-device"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode-postopt"
// CHECK-DEVICE-LIBS-SAME: "-Wno-linker-warnings"

// CHECK-NO-DEVICE-LIBS: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-NO-DEVICE-LIBS-SAME: "-fsycl-is-device"
// CHECK-NO-DEVICE-LIBS-NOT: "-mlink-builtin-bitcode"
// CHECK-NO-DEVICE-LIBS-NOT: "libsycl-crt.bc"
