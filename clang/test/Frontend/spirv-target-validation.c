// RUN: not %clang_cc1 -triple spirv-unknown-vulkan1.0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-VULKAN
// RUN: not %clang_cc1 -triple spirv-vulkan-mlibc %s 2>&1 | FileCheck %s --check-prefix=CHECK-SHADER
// RUN: %clang_cc1 -triple spirv %s -fsyntax-only 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-SPIRV-BARE
// RUN: %clang_cc1 -triple spirv32-mesa-mesa3d %s -fsyntax-only 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-SPIRV32-MESA
// RUN: %clang_cc1 -triple spirv64-mesa-mesa3d %s -fsyntax-only 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-SPIRV64-MESA
// RUN: %clang_cc1 -triple spirv-mesa-mesa3d %s -fsyntax-only 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-SPIRV-MESA

// CHECK-VULKAN: error: SPIR-V target requires a Vulkan environment
// CHECK-SHADER: error: SPIR-V target requires a valid shader stage or no environment
// CHECK-SPIRV-BARE-NOT: error: SPIR-V target requires a Vulkan environment
// CHECK-SPIRV32-MESA-NOT: error: unknown target triple 'spirv32-mesa-mesa3d'
// CHECK-SPIRV64-MESA-NOT: error: unknown target triple 'spirv64-mesa-mesa3d'
// CHECK-SPIRV-MESA-NOT: error: SPIR-V target requires a Vulkan environment

int main() { return 0; }
