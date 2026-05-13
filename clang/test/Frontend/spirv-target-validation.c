// RUN: not %clang_cc1 -triple spirv-unknown-vulkan1.0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-VULKAN
// RUN: not %clang_cc1 -triple spirv-vulkan-mlibc %s 2>&1 | FileCheck %s --check-prefix=CHECK-SHADER
// RUN: %clang_cc1 -triple spirv32-mesa-mesa3d %s -fsyntax-only 2>&1
// RUN: %clang_cc1 -triple spirv64-mesa-mesa3d %s -fsyntax-only 2>&1
// RUN: %clang_cc1 -triple spirv-mesa-mesa3d %s -fsyntax-only 2>&1

// CHECK-VULKAN: error: SPIR-V target requires a Vulkan environment
// CHECK-SHADER: error: SPIR-V target requires a valid shader stage or no environment

int main() { return 0; }
