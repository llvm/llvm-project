// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-offload.cir
// RUN: FileCheck --input-file=%t-offload.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

// Ordinary code compiled with -fopenmp.

int g;

// CIR: cir.global external @g = #cir.int<0> : !s32i
// LLVM: @g = global i32 0

struct S {
  int x;
};

void scalar_ops(S *s, int v) {
  g = v;
  g += v;
  ++g;
  s->x = g;
}

// CIR: cir.func{{.*}} @_Z10scalar_opsP1Si
// CIR:   %[[G:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %{{.*}}, %[[G]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[G1:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   %[[G1_VAL:.*]] = cir.load{{.*}} %[[G1]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[ADD:.*]] = cir.add nsw %[[G1_VAL]], %{{.*}} : !s32i
// CIR:   cir.store{{.*}} %[[ADD]], %[[G1]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[G2:.*]] = cir.get_global @g : !cir.ptr<!s32i>
// CIR:   %[[G2_VAL:.*]] = cir.load{{.*}} %[[G2]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[INC:.*]] = cir.inc nsw %[[G2_VAL]] : !s32i
// CIR:   cir.store{{.*}} %[[INC]], %[[G2]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[X:.*]] = cir.get_member %{{.*}}[0] {name = "x"} : !cir.ptr<!rec_S> -> !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %{{.*}}, %[[X]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z10scalar_opsP1Si
// LLVM:   store i32 %{{.*}}, ptr @g
// LLVM:   %[[G1:.*]] = load i32, ptr @g
// LLVM:   %[[ADD:.*]] = add nsw i32 %[[G1]], %{{.*}}
// LLVM:   store i32 %[[ADD]], ptr @g
// LLVM:   %[[G2:.*]] = load i32, ptr @g
// LLVM:   %[[INC:.*]] = add nsw i32 %[[G2]], 1
// LLVM:   store i32 %[[INC]], ptr @g
// LLVM:   %[[X:.*]] = getelementptr inbounds nuw %struct.S, ptr %{{.*}}, i32 0, i32 0
// LLVM:   store i32 %{{.*}}, ptr %[[X]]

void complex_ops(float _Complex *c, float _Complex v) {
  *c = v;
  ++*c;
}

// CIR: cir.func{{.*}} @_Z11complex_opsPCfS_
// CIR:   %[[V:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   %[[C_PTR:.*]] = cir.load deref{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!cir.complex<!cir.float>>>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   cir.store{{.*}} %[[V]], %[[C_PTR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[C_PTR2:.*]] = cir.load deref{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!cir.complex<!cir.float>>>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[C:.*]] = cir.load{{.*}} %[[C_PTR2]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   %[[REAL:.*]] = cir.complex.real %[[C]] : !cir.complex<!cir.float> -> !cir.float
// CIR:   %[[IMAG:.*]] = cir.complex.imag %[[C]] : !cir.complex<!cir.float> -> !cir.float
// CIR:   %[[INC:.*]] = cir.fadd %[[REAL]], %{{.*}} : !cir.float
// CIR:   %[[RES:.*]] = cir.complex.create %[[INC]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR:   cir.store{{.*}} %[[RES]], %{{.*}} : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVMCIR: define{{.*}} void @_Z11complex_opsPCfS_
// LLVMCIR:   store { float, float } %{{.*}}, ptr %{{.*}}
// LLVMCIR:   %[[REAL:.*]] = extractvalue { float, float } %{{.*}}, 0
// LLVMCIR:   %[[INC:.*]] = fadd float %[[REAL]], 1.000000e+00
// LLVMCIR:   store { float, float } %{{.*}}, ptr %{{.*}}

// OGCG: define{{.*}} void @_Z11complex_opsPCfS_
// OGCG:   store float %{{.*}}, ptr %{{.*}}
// OGCG:   store float %{{.*}}, ptr %{{.*}}
// OGCG:   %[[REAL:.*]] = load float, ptr %{{.*}}
// OGCG:   %[[INC:.*]] = fadd float %[[REAL]], 1.000000e+00
