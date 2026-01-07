// REQUIRES: directx-registered-target
// RUN: not %clang_cc1 -triple dxilv1.3-unknown-shadermodel6.3-library \
// RUN:  -finclude-default-header -S -o - %s 2>&1 | FileCheck %s

// CHECK: error: Unsupported intrinsic llvm.vector.reduce.and.v4i32 for DXIL lowering

export int vecReduceAndTest(int4 vec) {
    return __builtin_reduce_and(vec);
}
