// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,SHARED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=OGCG,SHARED

typedef bool v4b __attribute__((ext_vector_type(4)));
typedef bool v5b __attribute__((ext_vector_type(5)));
typedef bool v8b __attribute__((ext_vector_type(8)));

void vec_bool_without_padding_needed() {
  v8b a;
  v8b b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, {{.*}}align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, {{.*}}align 1

void vec_bool_load_store_without_padding_needed() {
  v8b a;
  v8b b;
  a = b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[TMP_B]], %[[A_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, {{.*}}align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, {{.*}}align 1
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[TMP_B_I8:.*]] = bitcast <8 x i1> %[[TMP_B_VEC]] to i8
// SHARED: store i8 %[[TMP_B_I8]], ptr %[[A_ADDR]], align 1

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

// SHARED: %[[A_ADDR:.*]] = alloca i8, {{.*}}align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, {{.*}}align 1
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[B_ELEM_3:.*]] = extractelement <8 x i1> %[[TMP_B_VEC]], i32 3
// SHARED: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// SHARED: %[[TMP_A_VEC:.*]] = bitcast i8 %[[TMP_A]] to <8 x i1>
// SHARED: %[[RESULT:.*]] = insertelement <8 x i1> %[[TMP_A_VEC]], i1 %[[B_ELEM_3]], i32 2
// SHARED: %[[RESULT_I8:.*]] = bitcast <8 x i1> %[[RESULT]] to i8
// SHARED: store i8 %[[RESULT_I8]], ptr %[[A_ADDR]], align 1

void vec_bool_4_extract_insert_without_padding_needed() {
  v4b a;
  v4b b;
  a[2] = b[3];
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.bool>>
// CIR: %1 = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.bool>>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.bool>>, !cir.vector<4 x !cir.bool>
// CIR: %[[CONST_3:.*]] = cir.const #cir.int<3> : !s32i
// CIR: %[[EXTRACT:.*]] = cir.vec.extract %[[TMP_B]][%[[CONST_3]] : !s32i] : !cir.vector<4 x !cir.bool>
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.bool>>, !cir.vector<4 x !cir.bool>
// CIR: %[[INSERT:.*]] = cir.vec.insert %[[EXTRACT]], %[[TMP_A]][%[[CONST_2]] : !s32i] : !cir.vector<4 x !cir.bool>
// CIR: cir.store {{.*}} %[[INSERT]], %[[A_ADDR]] : !cir.vector<4 x !cir.bool>, !cir.ptr<!cir.vector<4 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[EXTRACT_VEC:.*]] = shufflevector <8 x i1> %[[TMP_B_VEC]], <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// SHARED: %[[B_ELEM_3:.*]] = extractelement <4 x i1> %[[EXTRACT_VEC]], i32 3
// SHARED: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// SHARED: %[[TMP_A_VEC:.*]] = bitcast i8 %{{.*}} to <8 x i1>

// LLVM: %[[TMP_A_VEC_4:.*]] = shufflevector <8 x i1> %[[TMP_A_VEC]], <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// LLVM: %[[INSERT:.*]] = insertelement <4 x i1> %[[TMP_A_VEC_4]], i1 %[[B_ELEM_3]], i32 2
// LLVM: %[[TMP_A_VEC_8:.*]] = shufflevector <4 x i1> %[[INSERT]], <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>

// OGCG: %[[TMP_A_VEC_8:.*]] = insertelement <8 x i1> %[[TMP_A_VEC]], i1 %[[B_ELEM_3]], i32 2

// SHARED: %[[RESULT:.*]] = bitcast <8 x i1> %[[TMP_A_VEC_8]] to i8
// SHARED: store i8 %[[RESULT]], ptr %[[A_ADDR]], align 1

void vec_bool_load_store_with_padding_needed() {
  v4b a;
  v4b b;
  a = b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<4 x !cir.bool>>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<4 x !cir.bool>>, !cir.vector<4 x !cir.bool>
// CIR: cir.store {{.*}} %[[TMP_B]], %[[A_ADDR]] : !cir.vector<4 x !cir.bool>, !cir.ptr<!cir.vector<4 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, {{.*}}align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, {{.*}}align 1
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[EXTRACT_VEC:.*]] = shufflevector <8 x i1> %[[TMP_B_VEC]], <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// SHARED: %[[INSERT_VEC:.*]] = shufflevector <4 x i1> %[[EXTRACT_VEC]], <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
// SHARED: %[[RESULT:.*]] = bitcast <8 x i1> %[[INSERT_VEC]] to i8
// SHARED: store i8 %[[RESULT]], ptr %[[A_ADDR]], align 1

void vec_bool_5_load_store_with_padding_needed() {
  v5b a;
  v5b b;
  a = b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<5 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<5 x !cir.bool>>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<5 x !cir.bool>>, !cir.vector<5 x !cir.bool>
// CIR: cir.store {{.*}} %[[TMP_B]], %[[A_ADDR]] : !cir.vector<5 x !cir.bool>, !cir.ptr<!cir.vector<5 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[EXTRACT_VEC:.*]] = shufflevector <8 x i1> %[[TMP_B_VEC]], <8 x i1> poison, <5 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4>
// SHARED: %[[INSERT_VEC:.*]] = shufflevector <5 x i1> %[[EXTRACT_VEC]], <5 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 poison, i32 poison, i32 poison>
// SHARED: %[[RESULT:.*]] = bitcast <8 x i1> %[[INSERT_VEC]] to i8
// SHARED: store i8 %[[RESULT]], ptr %[[A_ADDR]], align 1
