// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

typedef int vi4 __attribute__((ext_vector_type(4)));
typedef int vi3 __attribute__((ext_vector_type(3)));
typedef int vi2 __attribute__((ext_vector_type(2)));
typedef double vd2 __attribute__((ext_vector_type(2)));

vi4 vec_a;
// CIR: cir.global external @[[VEC_A:.*]] = #cir.zero : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_A:.*]] = dso_local global <4 x i32> zeroinitializer

// OGCG: @[[VEC_A:.*]] = global <4 x i32> zeroinitializer

vi3 vec_b;
// CIR: cir.global external @[[VEC_B:.*]] = #cir.zero : !cir.vector<3 x !s32i>

// LLVM: @[[VEC_B:.*]] = dso_local global <3 x i32> zeroinitializer

// OGCG: @[[VEC_B:.*]] = global <3 x i32> zeroinitializer

vi2 vec_c;
// CIR: cir.global external @[[VEC_C:.*]] = #cir.zero : !cir.vector<2 x !s32i>

// LLVM: @[[VEC_C:.*]] = dso_local global <2 x i32> zeroinitializer

// OGCG: @[[VEC_C:.*]] = global <2 x i32> zeroinitializer

vd2 vec_d;

// CIR: cir.global external @[[VEC_D:.*]] = #cir.zero : !cir.vector<2 x !cir.double>

// LLVM: @[[VEC_D:.*]] = dso_local global <2 x double> zeroinitialize

// OGCG: @[[VEC_D:.*]] = global <2 x double> zeroinitializer

vi4 vec_e = { 1, 2, 3, 4 };

// CIR: cir.global external @[[VEC_E:.*]] = #cir.const_vector<[#cir.int<1> : !s32i, #cir.int<2> :
// CIR-SAME: !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.vector<4 x !s32i>

// LLVM: @[[VEC_E:.*]] = dso_local global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

// OGCG: @[[VEC_E:.*]] = global <4 x i32> <i32 1, i32 2, i32 3, i32 4>

void foo() {
  vi4 a;
  vi3 b;
  vi2 c;
  vd2 d;
}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[VEC_B:.*]] = cir.alloca !cir.vector<3 x !s32i>, !cir.ptr<!cir.vector<3 x !s32i>>, ["b"]
// CIR: %[[VEC_C:.*]] = cir.alloca !cir.vector<2 x !s32i>, !cir.ptr<!cir.vector<2 x !s32i>>, ["c"]
// CIR: %[[VEC_D:.*]] = cir.alloca !cir.vector<2 x !cir.double>, !cir.ptr<!cir.vector<2 x !cir.double>>, ["d"]

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[VEC_B:.*]] = alloca <3 x i32>, i64 1, align 16
// LLVM: %[[VEC_C:.*]] = alloca <2 x i32>, i64 1, align 8
// LLVM: %[[VEC_D:.*]] = alloca <2 x double>, i64 1, align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[VEC_B:.*]] = alloca <3 x i32>, align 16
// OGCG: %[[VEC_C:.*]] = alloca <2 x i32>, align 8
// OGCG: %[[VEC_D:.*]] = alloca <2 x double>, align 16

void foo2(vi4 p) {}

// CIR: %[[VEC_A:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["p", init]
// CIR: cir.store %{{.*}}, %[[VEC_A]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[VEC_A:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> %{{.*}}, ptr %[[VEC_A]], align 16

// OGCG: %[[VEC_A:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> %{{.*}}, ptr %[[VEC_A]], align 16
