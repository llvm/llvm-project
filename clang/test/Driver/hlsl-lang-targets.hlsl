// REQUIRES: directx-registered-target

// Supported targets
//
// RUN: %clang -target dxil--shadermodel6.2-pixel %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-VALID %s
// RUN: %clang -target dxil-unknown-shadermodel6.2-pixel %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-VALID %s
// RUN: %clang -target dxil--shadermodel6.2-library %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-VALID %s
// RUN: %clang -target dxil-unknown-shadermodel6.2-library %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-VALID %s

// Empty shader model
//
// RUN: not %clang -target dxil %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-OS %s

// Invalid shader models
//
// RUN: not %clang -target dxil--linux %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s
// RUN: not %clang -target dxil--win32 %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s
// RUN: not %clang -target dxil--unknown %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s
// RUN: not %clang -target dxil--invalidos %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s

// Bad shader model versions. Currently we just check for any version at all.
//
// RUN: not %clang -target dxil--shadermodel %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s
// RUN: not %clang -target dxil--shadermodel0.0 %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-OS %s

// Empty shader stage
//
// RUN: not %clang -target dxil-shadermodel6.2 %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-ENV %s
// RUN: not %clang -target dxil--shadermodel6.2 %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-ENV %s
// RUN: not %clang -target dxil--shadermodel6.2 %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-ENV %s

// Invalid shader stages
//
// RUN: not %clang -target dxil--shadermodel6.2-unknown %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-ENV %s
// RUN: not %clang -target dxil--shadermodel6.2-invalidenvironment %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-ENV %s
// RUN: not %clang -target dxil--shadermodel6.2-eabi %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-ENV %s
// RUN: not %clang -target dxil--shadermodel6.2-msvc %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-ENV %s

// Non-dxil targets
//
// RUN: not %clang -target x86_64-unknown-unknown %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-TARGET %s
// RUN: not %clang -target x86_64-linux %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-TARGET %s
// RUN: not %clang -target amdgcn %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-BAD-TARGET %s

// CHECK-VALID-NOT: error:
// CHECK-NO-OS: error: shader model is required as OS in target '{{.*}}' for HLSL code generation
// CHECK-BAD-OS: error: shader model '{{.*}}' in target '{{.*}}' is invalid for HLSL code generation
// CHECK-NO-ENV: error: shader stage is required as environment in target '{{.*}}' for HLSL code generation
// CHECK-BAD-ENV: error: shader stage '{{.*}}' in target '{{.*}}' is invalid for HLSL code generation
// CHECK-BAD-TARGET: error: HLSL code generation is unsupported for target '{{.*}}'

[shader("pixel")]
void main() {}
