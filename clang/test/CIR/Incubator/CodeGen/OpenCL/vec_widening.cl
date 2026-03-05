// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-cir -fclangir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-llvm -fclangir -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM

kernel void vec_widening(local const short3 *l_in, local short3 *l_out)
{
    *l_out = *l_in + (short3)1;
}

// CIR: [[PTR:%.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.vector<!s16i x 4>, lang_address_space(offload_local)>, !cir.vector<!s16i x 4>
// CIR: [[POISON:%.*]] = cir.const #cir.poison : !cir.vector<!s16i x 4>
// CIR: [[SHUFFLE:%.*]] = cir.vec.shuffle([[PTR]], [[POISON]] : !cir.vector<!s16i x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!s16i x 3>
// CIR: [[ONE_I32:%.*]] = cir.const #cir.int<1> : !s32i
// CIR: [[ONE_I16:%.*]] = cir.cast integral [[ONE_I32]] : !s32i -> !s16i
// CIR: [[SPLAT:%.*]] = cir.vec.splat [[ONE_I16]] : !s16i, !cir.vector<!s16i x 3>
// CIR: cir.binop(add, [[SHUFFLE]], [[SPLAT]]) : !cir.vector<!s16i x 3>

// LLVM: [[LOAD:%.*]] = load <4 x i16>, ptr addrspace(3) %{{.*}}, align 8
// LLVM: [[SHUF:%.*]] = shufflevector <4 x i16> [[LOAD]], <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// LLVM: add <3 x i16> [[SHUF]], splat (i16 1)

// OG-LLVM: [[LOAD:%.*]] = load <4 x i16>, ptr addrspace(3) %{{.*}}, align 8
// OG-LLVM: [[SHUF:%.*]] = shufflevector <4 x i16> [[LOAD]], <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// OG-LLVM: add <3 x i16> [[SHUF]], splat (i16 1)
