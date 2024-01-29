// RUN: %clang_dxc -T cs_6_0 -spirv -### %s 2>&1 | FileCheck %s

// CHECK: "-triple" "spirv-unknown-shadermodel6.0-compute"
// CHECK-SAME: "-x" "hlsl"
