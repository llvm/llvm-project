// REQUIRES: directx-registered-target
// RUN: not %clang_dxc -T lib_6_3 %s 2>&1 | FileCheck %s

// CHECK: error: Unsupported intrinsic llvm.vector.reduce.and.v4i32 for DXIL lowering

export int vecReduceAndTest(int4 vec) {
    return __builtin_reduce_and(vec);
}
