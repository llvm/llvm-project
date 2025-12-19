/// Check "-fsycl-is-device" and the default triple "spirv64-unknown-unknown"
//  is passed when compiling for device.

// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-triple" "spirv64-unknown-unknown" {{.*}} "-fsycl-is-device"

/// Check "-fsycl-is-device" and explicitly specified triple "nvptx" is
//  passed when compiling for device.

// RUN:   %clang -### -fsycl -fsycl-device-only --target=nvptx %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=TARGET
//
// TARGET: "-triple" "spirv32-unknown-unknown" "-aux-triple" "nvptx"

