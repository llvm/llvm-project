// RUN: not %clang -target x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=X86
// RUN: not %clang -target dxil-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=DXIL
// RUN: not %clang -target x86_64-unknown-shadermodel %s 2>&1 | FileCheck %s --check-prefix=SM
// RUN: not %clang -target spirv64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=SPIRV


// A completely unsupported target...
// X86: error: HLSL code generation is unsupported for target 'x86_64-unknown-unknown'

// Poorly specified targets
// DXIL: error: HLSL code generation is unsupported for target 'dxil-unknown-unknown'
// SM: error: HLSL code generation is unsupported for target 'x86_64-unknown-shadermodel'

// FIXME// SPIRV: error: HLSL code generation is unsupported for target 'spirv64-unknown-unknown'
