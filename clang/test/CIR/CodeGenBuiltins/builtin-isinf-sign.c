// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --input-file=%t.cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int test_float_isinf_sign(float x) {
  // CIR-LABEL: test_float_isinf_sign
  // CIR: %[[ARG:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!cir.float>, !cir.float
  // CIR: %[[IS_INF:.*]] = cir.is_fp_class %[[ARG]], fcInf : (!cir.float) -> !cir.bool
  // CIR: %[[IS_NEG:.*]] = cir.signbit %[[ARG]] : !cir.float -> !cir.bool
  // CIR: %[[C_0:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[C_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[C_m1:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR: %[[SIGN:.*]] = cir.select if %[[IS_NEG]] then %[[C_m1]] else %[[C_1]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: %[[RET:.*]] = cir.select if %[[IS_INF]] then %[[SIGN]] else %[[C_0]] : (!cir.bool, !s32i, !s32i) -> !s32i
  // CIR: cir.store %[[RET]], %{{.*}} : !s32i, !cir.ptr<!s32i>

  // LLVM-LABEL: test_float_isinf_sign
  // LLVM: %[[ARG:.*]] = load float, ptr %{{.*}}
  // LLVM: %[[IS_INF:.*]] = call i1 @llvm.is.fpclass.f32(float %[[ARG]], i32 516)
  // LLVM: %[[BITCAST:.*]] = bitcast float %[[ARG]] to i32
  // LLVM: %[[IS_NEG:.*]] = icmp slt i32 %[[BITCAST]], 0
  // LLVM: %[[SIGN:.*]] = select i1 %[[IS_NEG]], i32 -1, i32 1
  // LLVM: %[[RET:.*]] = select i1 %[[IS_INF]], i32 %[[SIGN]], i32 0
  // LLVM: store i32 %[[RET]], ptr %{{.*}}, align 4

  // OGCG-LABEL: test_float_isinf_sign
  // OGCG: %[[ARG:.*]] = load float, ptr %{{.*}}
  // OGCG: %[[ABS:.*]] = call float @llvm.fabs.f32(float %[[ARG]])
  // OGCG: %[[IS_INF:.*]] = fcmp oeq float %[[ABS]], 0x7FF0000000000000
  // OGCG: %[[BITCAST:.*]] = bitcast float %[[ARG]] to i32
  // OGCG: %[[IS_NEG:.*]] = icmp slt i32 %[[BITCAST]], 0
  // OGCG: %[[SIGN:.*]] = select i1 %[[IS_NEG]], i32 -1, i32 1
  // OGCG: %[[RET:.*]] = select i1 %[[IS_INF]], i32 %[[SIGN]], i32 0
  // OGCG: ret i32 %[[RET]]
  return __builtin_isinf_sign(x);
}
