// RUN: not %clang -target spirv %s 2>&1 | FileCheck %s
// CHECK: error: SPIR-V target requires a Vulkan environment

int main() { return 0; }