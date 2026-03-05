// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-cir -fclangir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O2 -emit-llvm -fclangir -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM

int test_scalar(int val, char n) {
  return val >> (n & 0x1f);
}

int2 test_vec2(int2 val, char2 n) {
  return (int2)(test_scalar(val.x, n.x), test_scalar(val.y, n.y));
}

int3 test_vec3(int3 val, char3 n) {
  return (int3)(test_vec2(val.xy, n.xy), test_scalar(val.z, n.z));
}

// CIR-LABEL: cir.func no_inline optnone @test_vec3
// CIR: %[[IDX0:.*]] = cir.const #cir.int<0> : !u32i
// CIR: %[[E0:.*]] = cir.vec.extract %{{.*}}[%[[IDX0]] : !u32i] : !cir.vector<!s32i x 2>
// CIR: %[[IDX1:.*]] = cir.const #cir.int<1> : !u32i
// CIR: %[[E1:.*]] = cir.vec.extract %{{.*}}[%[[IDX1]] : !u32i] : !cir.vector<!s32i x 2>
// CIR: %[[V3:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<!s32i x 3>{{.*}}>, !cir.vector<!s32i x 3>
// CIR: %[[IDX2:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[VAL2:.*]] = cir.vec.extract %[[V3]]
// CIR: %[[N3:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vector<!s8i x 3>{{.*}}>, !cir.vector<!s8i x 3>
// CIR: %[[NIDX2:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[NVAL2:.*]] = cir.vec.extract %[[N3]]
// CIR: %[[SCALAR:.*]] = cir.call @test_scalar(%[[VAL2]], %[[NVAL2]])
// CIR: cir.vec.create(%[[E0]], %[[E1]], %[[SCALAR]] : !s32i, !s32i, !s32i) : !cir.vector<!s32i x 3>

// LLVM-LABEL: define spir_func <3 x i32> @test_vec3
// LLVM: %[[V0:.*]] = insertelement <3 x i32> poison, i32 %{{.*}}, i64 0
// LLVM: %[[V1:.*]] = insertelement <3 x i32> %[[V0]], i32 %{{.*}}, i64 1
// LLVM: %[[V2:.*]] = insertelement <3 x i32> %[[V1]], i32 %{{.*}}, i64 2
// LLVM: ret <3 x i32> %[[V2]]

// OG-LLVM-LABEL: define spir_func <3 x i32> @test_vec3
// OG-LLVM: %[[V0:.*]] = insertelement <3 x i32> poison, i32 %{{.*}}, i64 0
// OG-LLVM: %[[V1:.*]] = insertelement <3 x i32> %[[V0]], i32 %{{.*}}, i64 1
// OG-LLVM: %[[V2:.*]] = insertelement <3 x i32> %[[V1]], i32 %{{.*}}, i64 2
// OG-LLVM: ret <3 x i32> %[[V2]]