// RUN: not %clang_cc1 -target spirv %s 2>&1 | FileCheck %s
// CHECK: error: SPIR-V target requires a Vulkan environment (e.g. '-target spirv64-unknown-vulkan1.3')

int main() { return 0; }
