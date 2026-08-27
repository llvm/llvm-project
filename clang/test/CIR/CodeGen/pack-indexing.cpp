// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2c -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2c -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2c -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

auto pack_indexing(auto... p) { return p...[0]; }

// OGCG-DAG: %[[P_0:.*]] = alloca i32, align 4
// OGCG-DAG: %[[P_1:.*]] = alloca i32, align 4
// OGCG-DAG: %[[P_2:.*]] = alloca i32, align 4
// OGCG-DAG: %[[RESULT:.*]] = load i32, ptr %[[P_0]], align 4
// OGCG-DAG-NEXT: ret i32 %[[RESULT]]

int pack_indexing_scalar() { return pack_indexing(1, 2, 3); }

// CIR: %[[RET_VAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR: %[[RESULT:.*]] = cir.call @_Z13pack_indexingIJiiiEEDaDpT_({{.*}}, {{.*}}, {{.*}}) : (!s32i {llvm.noundef}, !s32i {llvm.noundef}, !s32i {llvm.noundef}) -> (!s32i {llvm.noundef})
// CIR: cir.store %[[RESULT]], %[[RET_VAL]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[RET_VAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP]] : !s32i

// CIR: %[[P_0:.*]] = cir.alloca "p" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[P_1:.*]] = cir.alloca "p" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[P_2:.*]] = cir.alloca "p" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[RET_VAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR: %[[RESULT:.*]] = cir.load{{.*}} %[[P_0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[RESULT]], %[[RET_VAL]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[RET_VAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP]] : !s32i

// LLVM: %[[RET_VAL:.*]] = alloca i32, align 4
// LLVM: %[[RESULT:.*]] = call noundef i32 @_Z13pack_indexingIJiiiEEDaDpT_(i32 noundef 1, i32 noundef 2, i32 noundef 3)
// LLVM: store i32 %[[RESULT]], ptr %[[RET_VAL]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[RET_VAL]], align 4
// LLVM: ret i32 %[[TMP]]

// LLVM: %[[P_0:.*]] = alloca i32, align 4
// LLVM: %[[P_1:.*]] = alloca i32, align 4
// LLVM: %[[P_2:.*]] = alloca i32, align 4
// LLVM: %[[RET_VAL:.*]] = alloca i32, align 4
// LLVM: %[[RESULT:.*]] = load i32, ptr %[[P_0]], align 4
// LLVM: store i32 %[[RESULT]], ptr %[[RET_VAL]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[RET_VAL]], align 4
// LLVM: ret i32 %[[TMP]]

// OGCG-DAG: %[[CALL:.*]] = call noundef i32 @_Z13pack_indexingIJiiiEEDaDpT_(i32 noundef 1, i32 noundef 2, i32 noundef 3)
// OGCG-DAG-NEXT: ret i32 %[[RESULT]]

float _Complex pack_indexing_complex() {
  return pack_indexing(__builtin_complex(1.0f, 2.0f),
                       __builtin_complex(3.0f, 4.0f));
}

// CIR: cir.func {{.*}} @_Z21pack_indexing_complexv()
// CIR:   %[[RET_VAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[COMPLEX_0:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[COMPLEX_1:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[CONST_COMPLEX_0:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR:   %[[CONST_COMPLEX_1:.*]] = cir.const #cir.const_complex<#cir.fp<3.000000e+00> : !cir.float, #cir.fp<4.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR:   cir.store {{.*}} %[[CONST_COMPLEX_0]], %[[COMPLEX_0]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[TMP_COMPLEX_0:.*]] = cir.load {{.*}} %[[COMPLEX_0]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.store {{.*}} %[[CONST_COMPLEX_1]], %[[COMPLEX_1]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[TMP_COMPLEX_1:.*]] = cir.load {{.*}} %[[COMPLEX_1]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.store %[[TMP_COMPLEX_0]], %[[A0:.*]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[A0_PTR:.*]] = cir.cast bitcast %[[A0]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR:   %[[A0_VEC:.*]] = cir.load %[[A0_PTR]] : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>
// CIR:   cir.store %[[TMP_COMPLEX_1]], %[[A1:.*]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[A1_PTR:.*]] = cir.cast bitcast %[[A1]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR:   %[[A1_VEC:.*]] = cir.load %[[A1_PTR]] : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>
// CIR:   %[[RESULT:.*]] = cir.call @_Z13pack_indexingIJCfS0_EEDaDpT_(%[[A0_VEC]], %[[A1_VEC]]) : (!cir.vector<2 x !cir.float> {llvm.noundef}, !cir.vector<2 x !cir.float> {llvm.noundef}) -> (!cir.vector<2 x !cir.float> {llvm.noundef})
// CIR:   cir.store %[[RESULT]], %[[R_SLOT:.*]] : !cir.vector<2 x !cir.float>, !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR:   %[[R_PTR:.*]] = cir.cast bitcast %[[R_SLOT]] : !cir.ptr<!cir.vector<2 x !cir.float>> -> !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[R_CPLX:.*]] = cir.load %[[R_PTR]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.store {{.*}} %[[R_CPLX]], %[[RET_VAL]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[TMP_RET:.*]] = cir.load %[[RET_VAL]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR:   cir.store %[[TMP_RET]], %[[RET_SLOT:.*]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[RET_PTR:.*]] = cir.cast bitcast %[[RET_SLOT]] : !cir.ptr<!cir.complex<!cir.float>> -> !cir.ptr<!cir.vector<2 x !cir.float>>
// CIR:   %[[RET_VEC:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>
// CIR:   cir.return %[[RET_VEC]] : !cir.vector<2 x !cir.float>

// LLVM: define {{.*}} <2 x float> @_Z21pack_indexing_complexv()
// LLVM:   %[[RET_VAL:.*]] = alloca { float, float }, align 4
// LLVM:   %[[COMPLEX_0:.*]] = alloca { float, float }, align 4
// LLVM:   %[[COMPLEX_1:.*]] = alloca { float, float }, align 4
// LLVM:   store { float, float } { float 1.000000e+00, float 2.000000e+00 }, ptr %[[COMPLEX_0]], align 4
// LLVM:   %[[TMP_COMPLEX_0:.*]] = load { float, float }, ptr %[[COMPLEX_0]], align 4
// LLVM:   store { float, float } { float 3.000000e+00, float 4.000000e+00 }, ptr %[[COMPLEX_1]], align 4
// LLVM:   %[[TMP_COMPLEX_1:.*]] = load { float, float }, ptr %[[COMPLEX_1]], align 4
// LLVM:   store { float, float } %[[TMP_COMPLEX_0]], ptr %[[A0:.*]], align 4
// LLVM:   %[[A0_VEC:.*]] = load <2 x float>, ptr %[[A0]], align 8
// LLVM:   store { float, float } %[[TMP_COMPLEX_1]], ptr %[[A1:.*]], align 4
// LLVM:   %[[A1_VEC:.*]] = load <2 x float>, ptr %[[A1]], align 8
// LLVM:   %[[RESULT:.*]] = call noundef <2 x float> @_Z13pack_indexingIJCfS0_EEDaDpT_(<2 x float> noundef %[[A0_VEC]], <2 x float> noundef %[[A1_VEC]])
// LLVM:   store <2 x float> %[[RESULT]], ptr %[[R_SLOT:.*]], align 8
// LLVM:   %[[R_CPLX:.*]] = load { float, float }, ptr %[[R_SLOT]], align 4
// LLVM:   store { float, float } %[[R_CPLX]], ptr %[[RET_VAL]], align 4
// LLVM:   %[[TMP_RET:.*]] = load { float, float }, ptr %[[RET_VAL]], align 4
// LLVM:   store { float, float } %[[TMP_RET]], ptr %[[RET_SLOT:.*]], align 4
// LLVM:   %[[RET_VEC:.*]] = load <2 x float>, ptr %[[RET_SLOT]], align 8
// LLVM:   ret <2 x float> %[[RET_VEC]]

// OGCG: define {{.*}} <2 x float> @_Z21pack_indexing_complexv()
// OGCG:   %[[RET_VAL:.*]] = alloca { float, float }, align 4
// OGCG:   %[[COMPLEX_0:.*]] = alloca { float, float }, align 4
// OGCG:   %[[COMPLEX_1:.*]] = alloca { float, float }, align 4
// OGCG:   %[[RESULT_ADDR:.*]] = alloca { float, float }, align 4
// OGCG:   %[[COMPLEX_0_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_0]], i32 0, i32 0
// OGCG:   %[[COMPLEX_0_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_0]], i32 0, i32 1
// OGCG:   store float 1.000000e+00, ptr %[[COMPLEX_0_REAL_PTR]], align 4
// OGCG:   store float 2.000000e+00, ptr %[[COMPLEX_0_IMAG_PTR]], align 4
// OGCG:   %[[TMP_COMPLEX_0:.*]] = load <2 x float>, ptr %[[COMPLEX_0]], align 4
// OGCG:   %[[COMPLEX_1_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_1]], i32 0, i32 0
// OGCG:   %[[COMPLEX_1_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[COMPLEX_1]], i32 0, i32 1
// OGCG:   store float 3.000000e+00, ptr %[[COMPLEX_1_REAL_PTR]], align 4
// OGCG:   store float 4.000000e+00, ptr %[[COMPLEX_1_IMAG_PTR]], align 4
// OGCG:   %[[TMP_COMPLEX_1:.*]] = load <2 x float>, ptr %[[COMPLEX_1]], align 4
// OGCG:   %[[RESULT:.*]] = call noundef <2 x float> @_Z13pack_indexingIJCfS0_EEDaDpT_(<2 x float> noundef %[[TMP_COMPLEX_0]], <2 x float> noundef %[[TMP_COMPLEX_1]])
// OGCG:   store <2 x float> %[[RESULT]], ptr %[[RESULT_ADDR]], align 4
// OGCG:   %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT_ADDR]], i32 0, i32 0
// OGCG:   %[[RESULT_REAL:.*]] = load float, ptr %[[RESULT_REAL_PTR]], align 4
// OGCG:   %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RESULT_ADDR]], i32 0, i32 1
// OGCG:   %[[RESULT_IMAG:.*]] = load float, ptr %[[RESULT_IMAG_PTR]], align 4
// OGCG:   %[[RET_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RET_VAL]], i32 0, i32 0
// OGCG:   %[[RET_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[RET_VAL]], i32 0, i32 1
// OGCG:   store float %[[RESULT_REAL]], ptr %[[RET_REAL_PTR]], align 4
// OGCG:   store float %[[RESULT_IMAG]], ptr %[[RET_IMAG_PTR]], align 4
// OGCG:   %[[TMP_RET:.*]] = load <2 x float>, ptr %[[RET_VAL]], align 4
// OGCG:   ret <2 x float> %[[TMP_RET]]
