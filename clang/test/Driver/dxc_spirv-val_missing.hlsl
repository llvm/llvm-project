// UNSUPPORTED: spirv-val
//
// Verify that a warning is emitted exactly once when spirv-val is not found.
// This test is unsupported when spirv-val is in the build directory because
// GetProgramPath finds it via the application directory search, bypassing PATH.

// RUN: env PATH="" %clang_dxc -spirv -I test -Tlib_6_3 -Fo %t.spv -### %s 2>&1 | FileCheck %s

// CHECK: spirv-val not found; resulting SPIR-V will not be validated
// CHECK-NOT: spirv-val not found; resulting SPIR-V will not be validated

// Also verify -Vd suppresses the warning.
// RUN: %clang_dxc -spirv -I test -Vd -Tlib_6_3 -### %s 2>&1 | FileCheck %s --check-prefix=VD
// VD: "-cc1"{{.*}}"-triple" "spirv1.6-unknown-vulkan1.3-library"
// VD-NOT: spirv-val not found; resulting SPIR-V will not be validated
