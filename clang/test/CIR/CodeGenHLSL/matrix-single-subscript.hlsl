// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.cir.ll %s
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef float matrix3x3 __attribute__((matrix_type(3, 3)));
typedef float vector3 __attribute__((ext_vector_type(3)));

void matrix_single_subscript() {
  matrix3x3 matrix;
  vector3 vec = matrix[1];
}

// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>
// CIR: %[[VEC_ADDR:.*]] = cir.alloca "vec" {{.*}} init : !cir.ptr<!cir.vector<3 x !cir.float>>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[TMP_MATRIX:.*]] = cir.load {{.*}} %[[MATRIX_ADDR]] : !cir.ptr<!cir.matrix<3 x 3 x !cir.float>>, !cir.matrix<3 x 3 x !cir.float>
// CIR: %[[VEC:.*]] = cir.matrix.extract_row %[[TMP_MATRIX]][%[[CONST_1]] : !s32i] : !cir.matrix<3 x 3 x !cir.float>
// CIR: cir.store {{.*}} %[[VEC]], %[[VEC_ADDR]] : !cir.vector<3 x !cir.float>, !cir.ptr<!cir.vector<3 x !cir.float>>

// LLVM: %[[MATRIX_ADDR:.*]] = alloca [3 x <3 x float>], align 4
// LLVM: %[[VEC_ADDR:.*]] = alloca <3 x float>, align 4
// LLVM: %[[TMP_MATRIX:.*]] = load <9 x float>, ptr %[[MATRIX_ADDR]], align 4
// LLVM: %[[EXTRACT_1:.*]] = extractelement <9 x float> %[[TMP_MATRIX]], i64 1
// LLVM: %[[INSERT_ROW_0:.*]] = insertelement <3 x float> {{.*}}, float %[[EXTRACT_1]], i32 0
// LLVM: %[[EXTRACT_4:.*]] = extractelement <9 x float> %[[TMP_MATRIX]], i64 4
// LLVM: %[[INSERT_ROW_1:.*]] = insertelement <3 x float> %[[INSERT_ROW_0]], float %[[EXTRACT_4]], i32 1
// LLVM: %[[EXTRACT_7:.*]] = extractelement <9 x float> %[[TMP_MATRIX]], i64 7
// LLVM: %[[INSERT_ROW_2:.*]] = insertelement <3 x float> %[[INSERT_ROW_1]], float %[[EXTRACT_7]], i32 2
// LLVM: store <3 x float> %[[INSERT_ROW_2]], ptr %[[VEC_ADDR]], align 4
