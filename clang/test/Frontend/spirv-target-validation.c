// RUN: not %clang -target spirv %s 2>&1 | FileCheck %s --check-prefix=CHECK-VULKAN
// RUN: not %clang -target spirv-unknown-vulkan1.3 %s 2>&1 | FileCheck %s --check-prefix=CHECK-SHADER

// CHECK-VULKAN: error: SPIR-V target requires a Vulkan environment
// CHECK-SHADER: error: SPIR-V target requires a valid shader stage environment

int main() { return 0; }
