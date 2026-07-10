// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fclangir -emit-llvm  %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

typedef short short3 __attribute__((ext_vector_type(3)));

void copy_short3(short3 *dst, const short3 *src) { *dst = *src; }

// CIR-LABEL: cir.func{{.*}} @copy_short3
// CIR: %[[LOAD:.*]] = cir.load{{.*}} : !cir.ptr<!cir.vector<4 x !s16i>>, !cir.vector<4 x !s16i>
// CIR: %[[NARROW:.*]] = cir.vec.shuffle(%[[LOAD]], {{.*}} : !cir.vector<4 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<3 x !s16i>
// CIR: %[[UNDEF:.*]] = cir.const #cir.undef : !cir.vector<3 x !s16i>
// CIR: %[[WIDE:.*]] = cir.vec.shuffle(%[[NARROW]], %[[UNDEF]] : !cir.vector<3 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s16i>
// CIR: cir.store{{.*}} %[[WIDE]], {{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>>

// LLVM-LABEL: define{{.*}} void @copy_short3
// LLVM: %[[LOAD:.*]] = load <4 x i16>
// LLVM: %[[NARROW:.*]] = shufflevector <4 x i16> %[[LOAD]], <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// LLVM: %[[WIDE:.*]] = shufflevector <3 x i16> %[[NARROW]], <3 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// LLVM: store <4 x i16> %[[WIDE]]
