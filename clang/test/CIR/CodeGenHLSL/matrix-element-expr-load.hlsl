// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.cir.ll %s
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef int matrix4x4 __attribute__((matrix_type(4, 4)));

void matrix_element_expr() {
    matrix4x4 matrix;
    int element =  matrix._11;
}

// CIR: %[[MATRIX_ADDR:.*]] = cir.alloca "matrix" {{.*}} : !cir.ptr<!cir.matrix<4 x 4 x !s32i>>
// CIR: %[[ELEMENT_ADDR:.*]] = cir.alloca "element" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[TMP_MATRIX:.*]] = cir.load {{.*}} %[[MATRIX_ADDR]] : !cir.ptr<!cir.matrix<4 x 4 x !s32i>>, !cir.matrix<4 x 4 x !s32i>
// CIR: %[[ROW_IDX:.*]] = cir.const #cir.int<0> : !s64i
// CIR: %[[COLUMN_IDX:.*]] = cir.const #cir.int<0> : !s64i
// CIR: %[[EXTREACT:.*]] = cir.matrix.extract %[[TMP_MATRIX]][%[[ROW_IDX]] : !s64i] [%[[COLUMN_IDX]] : !s64i] : !cir.matrix<4 x 4 x !s32i>
// CIR: cir.store {{.*}} %[[EXTREACT]], %[[ELEMENT_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[MATRIX_ADDR:.*]] = alloca [4 x <4 x i32>], align 4
// LLVM: %[[ELEMENT_ADDR:.*]] = alloca i32, align 4
// LLVM: %[[TMP_MATRIX:.*]] = load <16 x i32>, ptr %[[MATRIX_ADDR]], align 4
// LLVM: %[[EXTRACT:.*]] = extractelement <16 x i32> %[[TMP_MATRIX]], i64 0
// LLVM: store i32 %[[EXTRACT]], ptr %[[ELEMENT_ADDR]], align 4
