// RUN: not %clang_cc1 -triple spirv %s 2>&1 | FileCheck %s --check-prefix=CHECK-VULKAN
// RUN: not %clang_cc1 -triple spirv-vulkan-mlibc %s 2>&1 | FileCheck %s --check-prefix=CHECK-SHADER

// CHECK-VULKAN: error: SPIR-V target requires a Vulkan environment
// CHECK-SHADER: error: SPIR-V target requires a valid shader stage or no environment

int main() { return 0; }
