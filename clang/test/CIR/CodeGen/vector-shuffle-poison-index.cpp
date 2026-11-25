// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

// A shuffle of two constant vectors folds to a constant vector. A -1 index
// leaves a poison element in it, as in classic codegen.

typedef int vi4 __attribute__((vector_size(16)));
typedef float vf4 __attribute__((vector_size(16)));

// The -1 index keeps the initializer from being a constant expression, so the
// folded vector is stored by the dynamic initializer.
vi4 g_shuffle =
    __builtin_shufflevector(vi4{1, 2, 3, 4}, vi4{5, 6, 7, 8}, 4, -1, 6, -1);

// CIR-LABEL: cir.func {{.*}} @__cxx_global_var_init
// CIR: cir.const #cir.const_vector<[#cir.int<5> : !s32i, #cir.poison : !s32i, #cir.int<7> : !s32i, #cir.poison : !s32i]> : !cir.vector<4 x !s32i>

// LLVM-LABEL: define internal void @__cxx_global_var_init()
// LLVM: store <4 x i32> <i32 5, i32 poison, i32 7, i32 poison>, ptr @g_shuffle, align 16
// LLVM-NEXT: ret void

vi4 shuffle_int() {
  return __builtin_shufflevector(vi4{1, 2, 3, 4}, vi4{5, 6, 7, 8}, -1, 1, 2, 3);
}

// CIR-LABEL: cir.func {{.*}} @_Z11shuffle_intv
// CIR: cir.const #cir.const_vector<[#cir.poison : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.vector<4 x !s32i>

// LLVM-LABEL: define dso_local noundef <4 x i32> @_Z11shuffle_intv()
// FIXME: ClangIR stores the result to a return slot and reloads it.
// LLVMCIR: store <4 x i32> <i32 poison, i32 2, i32 3, i32 4>, ptr %[[SLOT:.*]], align 16
// LLVMCIR-NEXT: %[[RET:.*]] = load <4 x i32>, ptr %[[SLOT]], align 16
// LLVMCIR-NEXT: ret <4 x i32> %[[RET]]
// OGCG: ret <4 x i32> <i32 poison, i32 2, i32 3, i32 4>

vf4 shuffle_float() {
  return __builtin_shufflevector(vf4{1.0f, 2.0f, 3.0f, 4.0f},
                                 vf4{5.0f, 6.0f, 7.0f, 8.0f}, 0, -1, 6, -1);
}

// CIR-LABEL: cir.func {{.*}} @_Z13shuffle_floatv
// CIR: cir.const #cir.const_vector<[#cir.fp<1.000000e+00> : !cir.float, #cir.poison : !cir.float, #cir.fp<7.000000e+00> : !cir.float, #cir.poison : !cir.float]> : !cir.vector<4 x !cir.float>

// LLVM-LABEL: define dso_local noundef <4 x float> @_Z13shuffle_floatv()
// FIXME: ClangIR stores the result to a return slot and reloads it.
// LLVMCIR: store <4 x float> <float 1.000000e+00, float poison, float 7.000000e+00, float poison>, ptr %[[SLOT:.*]], align 16
// LLVMCIR-NEXT: %[[RET:.*]] = load <4 x float>, ptr %[[SLOT]], align 16
// LLVMCIR-NEXT: ret <4 x float> %[[RET]]
// OGCG: ret <4 x float> <float 1.000000e+00, float poison, float 7.000000e+00, float poison>
