// These are placeholder errors since we do not currently support -Fre to
// generate reflection data for DXIL or SPIRV targets.

// RUN: env PATH="" not %clang_dxc -T cs_6_0 %s -Fre blah.json -Vd -### 2>&1 | FileCheck --check-prefix=FRE_DXIL %s
// FRE_DXIL: error: unsupported option '-Fre' for target 'dxilv1.0'

// RUN: env PATH="" not %clang_dxc -T cs_6_0 %s -spirv -Fre blah.json -Vd -### 2>&1 | FileCheck --check-prefix=FRE_SPV %s
// FRE_SPV: error: unsupported option '-Fre' for target 'spirv1.6'
