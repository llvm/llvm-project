// RUN: not %clang_cc1 -triple amdgcn -target-feature +wavefrontsize32 -target-feature +wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple amdgcn -target-cpu gfx1103 -target-feature +wavefrontsize32 -target-feature +wavefrontsize64 -o /dev/null %s 2>&1 | FileCheck %s

// CHECK: error: invalid feature combination: 'wavefrontsize32' and 'wavefrontsize64' are mutually exclusive

kernel void test() {}
