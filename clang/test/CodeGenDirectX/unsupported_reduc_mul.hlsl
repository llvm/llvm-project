// RUN: %if clang-dxc %{not %clang_dxc -T lib_6_3 %s 2>&1 | FileCheck %s %}

// CHECK: error: <unknown>:0:0: in function llvm.vector.reduce.mul.v4i32 i32 (<4 x i32>): Unsupported intrinsic for DXIL lowering

export int vecReduceMulTest(int4 vec) {
    return __builtin_reduce_mul(vec);
}
