// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef int vi4 __attribute__((vector_size(16)));
typedef double vd2 __attribute__((vector_size(16)));
typedef long long vll2 __attribute__((vector_size(16)));

vi4 vec_a;
// CIR: cir.global external @[[VEC_A:.*]] = #cir.zero : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_A:.*]] = dso_local global <4 x i32> zeroinitializer

// OGCG: @[[VEC_A:.*]] = global <4 x i32> zeroinitializer

vd2 b;
// CIR: cir.global external @[[VEC_B:.*]] = #cir.zero : !cir.vector<2 x !cir.double>

// LLVM: @[[VEC_B:.*]] = dso_local global <2 x double> zeroinitialize

// OGCG: @[[VEC_B:.*]] = global <2 x double> zeroinitializer

vll2 c;
// CIR: cir.global external @[[VEC_C:.*]] = #cir.zero : !cir.vector<2 x !s64i>

// LLVM: @[[VEC_C:.*]] = dso_local global <2 x i64> zeroinitialize

// OGCG: @[[VEC_C:.*]] = global <2 x i64> zeroinitializer

vi4 d = { 1, 2, 3, 4 };

// CIR: cir.global external @[[VEC_D:.*]] = #cir.const_vector<[#cir.int<1> : !s32i, #cir.int<2> :
// CIR-SAME: !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_D:.*]] = dso_local global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

// OGCG: @[[VEC_D:.*]] = global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

int x = 5;

void foo() {
  vi4 a;
  vd2 b;
  vll2 c;

  vi4 d = { 1, 2, 3, 4 };

  vi4 e = { x, 5, 6, x + 1 };

  vi4 f = { 5 };
}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[VEC_B:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["b"]
// CIR: %[[VEC_C:.*]] = cir.alloca !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>, ["c"]
// CIR: %[[VEC_D:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["d", init]
// CIR: %[[VEC_E:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["e", init]
// CIR: %[[VEC_F:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["f", init]
// CIR: %[[VEC_D_VAL:.*]] = cir.vec.create({{.*}}, {{.*}}, {{.*}}, {{.*}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_D_VAL]], %[[VEC_D]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[VEC_E_VAL:.*]] = cir.vec.create({{.*}}, {{.*}}, {{.*}}, {{.*}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_E_VAL]], %[[VEC_E]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[VEC_F_VAL:.*]] = cir.vec.create({{.*}}, {{.*}}, {{.*}}, {{.*}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<4 x !s32i>
// CIR: cir.store %[[VEC_F_VAL]], %[[VEC_F]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_B:.*]] = alloca <2 x double>, i64 1, align 16
// LLVM: %[[VEC_C:.*]] = alloca <2 x i64>, i64 1, align 16
// LLVM: %[[VEC_D:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_E:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_F:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_D]], align 16
// LLVM: store <4 x i32> {{.*}}, ptr %[[VEC_E]], align 16
// LLVM: store <4 x i32> <i32 5, i32 0, i32 0, i32 0>, ptr %[[VEC_F]], align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_B:.*]] = alloca <2 x double>, align 16
// OGCG: %[[VEC_C:.*]] = alloca <2 x i64>, align 16
// OGCG: %[[VEC_D:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_E:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_F:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %[[VEC_D]], align 16
// OGCG: store <4 x i32> {{.*}}, ptr %[[VEC_E]], align 16
// OGCG: store <4 x i32> <i32 5, i32 0, i32 0, i32 0>, ptr %[[VEC_F]], align 16

void foo2(vi4 p) {}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["p", init]
// CIR: cir.store %{{.*}}, %[[VEC_A]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> %{{.*}}, ptr %[[VEC_A]], align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> %{{.*}}, ptr %[[VEC_A]], align 16
