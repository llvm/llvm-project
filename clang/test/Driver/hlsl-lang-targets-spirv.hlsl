// REQUIRES: spirv-registered-target
// REQUIRES: directx-registered-target

// Supported targets
//
// RUN: %clang -target dxil-unknown-shadermodel6.2-compute %s -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=CHECK-VALID %s
// RUN: %clang -target spirv-unknown-vulkan-compute %s -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=CHECK-VALID %s
// RUN: %clang -target spirv-unknown-vulkan1.2-compute %s -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=CHECK-VALID %s
// RUN: %clang -target spirv-unknown-vulkan1.3-compute %s -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=CHECK-VALID %s
// RUN: %clang -target spirv1.5-unknown-vulkan1.2-compute %s -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=CHECK-VALID %s
// RUN: %clang -target spirv1.6-unknown-vulkan1.3-compute %s -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=CHECK-VALID %s

// Empty Vulkan environment
//
// RUN: not %clang -target spirv %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-OS %s

// Invalid Vulkan environment
//
// RUN: not %clang -target spirv--shadermodel %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s
// RUN: not %clang -target spirv-unknown-vulkan1.0-compute %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s
// RUN: not %clang -target spirv1.5-unknown-vulkan1.3-compute %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s

// Invalid SPIR-V version
// RUN: not %clang -target spirv1.0-unknown-vulkan-compute %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-TARGET %s

// Empty shader stage
//
// RUN: not %clang -target spirv--vulkan %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-ENV %s

// Invalid shader stages
//
// RUN: not %clang -target spirv--vulkan-unknown %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-ENV %s

// CHECK-VALID-NOT: error:
// CHECK-NO-OS: error: Vulkan environment is required as OS in target '{{.*}}' for HLSL code generation
// CHECK-BAD-OS: error: Vulkan environment '{{.*}}' in target '{{.*}}' is invalid for HLSL code generation
// CHECK-NO-ENV: error: shader stage is required as environment in target '{{.*}}' for HLSL code generation
// CHECK-BAD-ENV: error: shader stage '{{.*}}' in target '{{.*}}' is invalid for HLSL code generation
// CHECK-BAD-TARGET: error: HLSL code generation is unsupported for target '{{.*}}'

[shader("compute"), numthreads(1,1,1)]
void main() {}
