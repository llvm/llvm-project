// RUN: %clang_dxc -T cs_6_0 -spirv -### %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -T cs_6_0 -spirv -fspv-target-env=vulkan1.2 -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-VULKAN12
// RUN: %clang_dxc -T cs_6_0 -spirv -fspv-target-env=vulkan1.3 -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-VULKAN13
// RUN: not %clang_dxc -T cs_6_0 -spirv -fspv-target-env=vulkan1.0 -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// CHECK: "-triple" "spirv1.6-unknown-vulkan1.3-compute"
// CHECK-SAME: "-x" "hlsl"

// CHECK-VULKAN12: "-triple" "spirv1.5-unknown-vulkan1.2-compute"

// CHECK-VULKAN13: "-triple" "spirv1.6-unknown-vulkan1.3-compute"

// CHECK-ERROR: error: invalid value 'vulkan1.0' in '-fspv-target-env=vulkan1.0'
