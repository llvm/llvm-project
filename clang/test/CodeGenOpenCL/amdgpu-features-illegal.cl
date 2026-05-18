// RUN: not %clang_cc1 -triple amdgcn -target-feature +wavefrontsize32 -target-feature +wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck --check-prefix=INVALID-PLUS %s
// RUN: not %clang_cc1 -triple amdgcn -target-feature -wavefrontsize32 -target-feature -wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck --check-prefix=INVALID-MINUS %s
// RUN: not %clang_cc1 -triple amdgcn -target-cpu gfx1103 -target-feature +wavefrontsize32 -target-feature +wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck --check-prefix=INVALID-PLUS %s
// RUN: not %clang_cc1 -triple amdgcn -target-cpu gfx1103 -target-feature -wavefrontsize32 -target-feature -wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck --check-prefix=INVALID-MINUS %s
// RUN: not %clang_cc1 -triple amdgcn -target-cpu gfx900 -target-feature +wavefrontsize32 -target-feature -wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=GFX9-WAVE32
// RUN: not %clang_cc1 -triple amdgcn -target-cpu gfx900 -target-feature -wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=GFX9-WAVE64
// RUN: not %clang_cc1 -triple amdgcn -target-cpu gfx1250 -target-feature +wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=GFX1250-WAVE64
// RUN: not %clang_cc1 -triple amdgcn -target-cpu gfx1250 -target-feature -wavefrontsize32 -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=GFX1250-WAVE32

// INVALID-PLUS: error: invalid feature combination: '+wavefrontsize32' and '+wavefrontsize64' are mutually exclusive
// INVALID-MINUS: error: invalid feature combination: '-wavefrontsize32' and '-wavefrontsize64' are mutually exclusive
// GFX9-WAVE32: error: option '+wavefrontsize32' cannot be specified on this target
// GFX9-WAVE64: error: option '-wavefrontsize64' cannot be specified on this target
// GFX1250-WAVE64: error: option '+wavefrontsize64' cannot be specified on this target
// GFX1250-WAVE32: error: option '-wavefrontsize32' cannot be specified on this target

kernel void test() {}
