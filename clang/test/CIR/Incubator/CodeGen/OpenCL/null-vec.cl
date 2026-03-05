// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-cir -o - %s -fclangir | FileCheck %s --check-prefix=CIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-llvm -o - %s -fclangir | FileCheck %s --check-prefix=LLVM
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM

typedef __attribute__(( ext_vector_type(2) )) unsigned int uint2;

kernel void test_null_vec(uint2 in1, uint2 in2, local uint2 *out)
{
    uint2 tmp[2] = {0, 0};  // Vector of NULL vals

    if (in1.s0 != 1)
        tmp[0] = in1;
    if (in2.s1 != 2)
        tmp[1] = in2;
    *out = tmp[0] + tmp[1];
}

// CIR: cir.const #cir.zero : !cir.array<!cir.vector<!u32i x 2> x 2>
// CIR: cir.binop(add, %{{.*}}, %{{.*}}) : !cir.vector<!u32i x 2>
// LLVM: [[S1:%.*]] = select i1 %{{.*}}, <2 x i32> zeroinitializer, <2 x i32>
// LLVM: [[S2:%.*]] = select i1 %{{.*}}, <2 x i32> zeroinitializer, <2 x i32>
// LLVM: add <2 x i32> [[S2]], [[S1]]
// OG-LLVM: [[S1:%.*]] = select i1 %{{.*}}, <2 x i32> zeroinitializer, <2 x i32>
// OG-LLVM: [[S2:%.*]] = select i1 %{{.*}}, <2 x i32> zeroinitializer, <2 x i32>
// OG-LLVM: add <2 x i32> [[S2]], [[S1]]