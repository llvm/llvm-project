// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef bool v8b __attribute__((ext_vector_type(8)));

void vec_bool_without_padding_needed() {
  v8b a;
  v8b b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>

// LLVM: %[[A_ADDR:.*]] = alloca i8, {{.*}}align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, {{.*}}align 1

void vec_bool_load_store_without_padding_needed() {
  v8b a;
  v8b b;
  a = b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[TMP_B]], %[[A_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>

// LLVM: %[[A_ADDR:.*]] = alloca i8, {{.*}}align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, {{.*}}align 1
// LLVM: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// LLVM: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// LLVM: %[[TMP_B_I8:.*]] = bitcast <8 x i1> %[[TMP_B_VEC]] to i8
// LLVM: store i8 %[[TMP_B_I8]], ptr %[[A_ADDR]], align 1

void vec_bool_extract_insert_without_padding_needed() {
  v8b a;
  v8b b;
  a[2] = b[3];
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[B_ELEM_3:.*]] = cir.vec.extract %[[TMP_B]][%[[CONST_3]] : !s32i] : !cir.vector<8 x !cir.bool>
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[RESULT:.*]] = cir.vec.insert %[[B_ELEM_3]], %[[TMP_A]][%[[CONST_2]] : !s32i] : !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[RESULT]], %[[A_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>

// LLVM: %[[A_ADDR:.*]] = alloca i8, {{.*}}align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, {{.*}}align 1
// LLVM: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// LLVM: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// LLVM: %[[B_ELEM_3:.*]] = extractelement <8 x i1> %[[TMP_B_VEC]], i32 3
// LLVM: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// LLVM: %[[TMP_A_VEC:.*]] = bitcast i8 %[[TMP_A]] to <8 x i1>
// LLVM: %[[RESULT:.*]] = insertelement <8 x i1> %[[TMP_A_VEC]], i1 %[[B_ELEM_3]], i32 2
// LLVM: %[[RESULT_I8:.*]] = bitcast <8 x i1> %[[RESULT]] to i8
// LLVM: store i8 %[[RESULT_I8]], ptr %[[A_ADDR]], align 1
