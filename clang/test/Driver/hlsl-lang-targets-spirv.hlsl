// REQUIRES: spirv-registered-target

// Supported targets
//
// RUN: %clang -target dxil-unknown-shadermodel6.2-pixel %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-VALID %s
// RUN: %clang -target spirv-unknown-shadermodel6.2-library %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-VALID %s

// Empty shader model
//
// RUN: not %clang -target spirv %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-OS %s

// Invalid shader models
//
// RUN: not %clang -target spirv--unknown %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s

// Empty shader stage
//
// RUN: not %clang -target spirv--shadermodel6.2 %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-ENV %s

// Invalid shader stages
//
// RUN: not %clang -target spirv--shadermodel6.2-unknown %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-ENV %s

// CHECK-VALID-NOT: error:
// CHECK-NO-OS: error: shader model is required in target '{{.*}}' for HLSL code generation
// CHECK-BAD-OS: error: shader model '{{.*}}' in target '{{.*}}' is invalid for HLSL code generation
// CHECK-NO-ENV: error: shader stage is required in target '{{.*}}' for HLSL code generation
// CHECK-BAD-ENV: error: shader stage '{{.*}}' in target '{{.*}}' is invalid for HLSL code generation

[shader("pixel")]
void main() {}
