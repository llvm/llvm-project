// RUN: %clang_cc1 -triple spirv-unknown-vulkan -fclangir \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple spirv-unknown-vulkan -fclangir \
// RUN:   -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv-unknown-vulkan -emit-llvm \
// RUN:   -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=PHYSICAL-CIR
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir \
// RUN:   -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=PHYSICAL-LLVM
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm \
// RUN:   -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=PHYSICAL-LLVM

typedef short short3 __attribute__((ext_vector_type(3)));

void copy_short3(global short3 *dst, global const short3 *src) {
  *dst = *src;
}

// CIR-LABEL: cir.func{{.*}} @copy_short3
// CIR: %[[LOAD:.*]] = cir.load{{.*}} : !cir.ptr<!cir.vector<3 x !s16i>{{.*}}>, !cir.vector<3 x !s16i>
// CIR: cir.store{{.*}} %[[LOAD]], {{.*}} : !cir.vector<3 x !s16i>, !cir.ptr<!cir.vector<3 x !s16i>{{.*}}>

// LLVM-LABEL: define{{.*}} void @copy_short3
// LLVM: %[[LOAD:.*]] = load <3 x i16>
// LLVM: store <3 x i16> %[[LOAD]]

// PHYSICAL-CIR-LABEL: cir.func{{.*}} @copy_short3
// PHYSICAL-CIR: %[[LOAD:.*]] = cir.load{{.*}} : !cir.ptr<!cir.vector<4 x !s16i>{{.*}}>, !cir.vector<4 x !s16i>
// PHYSICAL-CIR: %[[NARROW:.*]] = cir.vec.shuffle(%[[LOAD]], {{.*}} : !cir.vector<4 x !s16i>) {{.*}} : !cir.vector<3 x !s16i>
// PHYSICAL-CIR: %[[UNDEF:.*]] = cir.const #cir.undef : !cir.vector<3 x !s16i>
// PHYSICAL-CIR: %[[WIDE:.*]] = cir.vec.shuffle(%[[NARROW]], %[[UNDEF]] : !cir.vector<3 x !s16i>) {{.*}} : !cir.vector<4 x !s16i>
// PHYSICAL-CIR: cir.store{{.*}} %[[WIDE]], {{.*}} : !cir.vector<4 x !s16i>, !cir.ptr<!cir.vector<4 x !s16i>{{.*}}>

// PHYSICAL-LLVM-LABEL: define{{.*}} void @copy_short3
// PHYSICAL-LLVM: %[[LOAD:.*]] = load <4 x i16>
// PHYSICAL-LLVM: %[[NARROW:.*]] = shufflevector <4 x i16> %[[LOAD]], <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// PHYSICAL-LLVM: %[[WIDE:.*]] = shufflevector <3 x i16> %[[NARROW]], <3 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// PHYSICAL-LLVM: store <4 x i16> %[[WIDE]]
