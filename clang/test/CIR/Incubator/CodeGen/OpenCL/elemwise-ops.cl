// RUN: %clang_cc1 %s -cl-std=CL2.0 -fclangir -emit-cir -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=CIR

// RUN: %clang_cc1 %s -cl-std=CL2.0 -fclangir -emit-llvm -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OG-LLVM

typedef __attribute__(( ext_vector_type(2) )) int int2;

// CIR: %[[LHS:.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s32i, #cir.int<3> : !s32i]> : !cir.vector<!s32i x 2>
// CIR: %[[WIDTH:.*]] = cir.const #cir.const_vector<[#cir.int<31> : !s32i, #cir.int<31> : !s32i]> : !cir.vector<!s32i x 2>
// CIR: %[[MASK:.*]] = cir.binop(and, %[[LHS]], %[[WIDTH]]) : !cir.vector<!s32i x 2>
// CIR: cir.shift(right, %{{.*}} : !cir.vector<!s32i x 2>, %[[MASK]] : !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>
// LLVM: ashr <2 x i32> %{{.*}}, splat (i32 3)
// OG-LLVM: ashr <2 x i32> %x, splat (i32 3)
int2 shr(int2 x)
{
    return x >> 3;
}

// CIR: %[[LHS:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[WIDTH:.*]] = cir.const #cir.int<31> : !s32i
// CIR: %[[MASK:.*]] = cir.binop(and, %[[LHS]], %[[WIDTH]]) : !s32i
// CIR: cir.shift(left, %{{.*}} : !s32i, %[[MASK]] : !s32i) -> !s32i
// LLVM: shl i16 %{{.*}}, 5
// OG-LLVM: shl i16 %x, 5
short shl(short x)
{
    return x << 5;
}