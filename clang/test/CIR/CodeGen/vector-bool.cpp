// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,SHARED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=OGCG,SHARED

typedef bool v4b __attribute__((ext_vector_type(4)));
typedef bool v5b __attribute__((ext_vector_type(5)));
typedef bool v8b __attribute__((ext_vector_type(8)));

void constant_vec_bool() {
    v8b a = {true, false, true, false, true, false, true, false};
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[CONST_VEC:.*]] = cir.const #cir.const_vector<[#true, #false, #true, #false, #true, #false, #true, #false]> : !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[CONST_VEC]], %[[A_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, align 1
// SHARED: store i8 85, ptr %[[A_ADDR]], align 1

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

void vec_bool_ternary_expr() {
  v8b a;
  v8b b;
  v8b c;
  v8b d = a ? b : c;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[D_ADDR:.*]] = cir.alloca "d" {{.*}} init : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[TMP_C:.*]] = cir.load {{.*}} %[[C_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[RESULT:.*]] = cir.vec.ternary(%[[TMP_A]], %[[TMP_B]], %[[TMP_C]]) : !cir.vector<8 x !cir.bool>, !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[RESULT]], %[[D_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[C_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[D_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// SHARED: %[[TMP_A_I8:.*]] = bitcast i8 %[[TMP_A]] to <8 x i1>
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_I8:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[TMP_C:.*]] = load i8, ptr %[[C_ADDR]], align 1
// SHARED: %[[TMP_C_I8:.*]] = bitcast i8 %[[TMP_C]] to <8 x i1>
// SHARED: %[[RESULT:.*]] = select <8 x i1> %[[TMP_A_I8]], <8 x i1> %[[TMP_B_I8]], <8 x i1> %[[TMP_C_I8]]
// SHARED: %[[RESULT_I8:.*]] = bitcast <8 x i1> %[[RESULT]] to i8
// SHARED: store i8 %[[RESULT_I8]], ptr %[[D_ADDR]], align 1

void vec_bool_bitwise_operators() {
  v8b a;
  v8b b;
  v8b v_or = a | b;
  v8b v_and = a & b;
  v8b v_xor = a ^ b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[OR_ADDR:.*]] = cir.alloca "v_or" {{.*}} init : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[AND_ADDR:.*]] = cir.alloca "v_and" {{.*}} init : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[XOR_ADDR:.*]] = cir.alloca "v_xor" {{.*}} init : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[OR:.*]] = cir.or %[[TMP_A]], %[[TMP_B]] : !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[OR]], %[[OR_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[AND:.*]] = cir.and %[[TMP_A]], %[[TMP_B]] : !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[AND]], %[[AND_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[TMP_B:.*]] = cir.load {{.*}} %[[B_ADDR]] : !cir.ptr<!cir.vector<8 x !cir.bool>>, !cir.vector<8 x !cir.bool>
// CIR: %[[XOR:.*]] = cir.xor %[[TMP_A]], %[[TMP_B]] : !cir.vector<8 x !cir.bool>
// CIR: cir.store {{.*}} %[[XOR]], %[[XOR_ADDR]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>

// SHARED: %[[A_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[B_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[OR_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[AND_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[XOR_ADDR:.*]] = alloca i8, align 1
// SHARED: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// SHARED: %[[TMP_A_VEC:.*]] = bitcast i8 %[[TMP_A]] to <8 x i1>
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[OR:.*]] = or <8 x i1> %[[TMP_A_VEC]], %[[TMP_B_VEC]]
// SHARED: %[[OR_I8:.*]] = bitcast <8 x i1> %[[OR]] to i8
// SHARED: store i8 %[[OR_I8]], ptr %[[OR_ADDR]], align 1
// SHARED: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// SHARED: %[[TMP_A_VEC:.*]] = bitcast i8 %[[TMP_A]] to <8 x i1>
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[AND:.*]] = and <8 x i1> %[[TMP_A_VEC]], %[[TMP_B_VEC]]
// SHARED: %[[AND_I8:.*]] = bitcast <8 x i1> %[[AND]] to i8
// SHARED: store i8 %[[AND_I8]], ptr %[[AND_ADDR]], align 1
// SHARED: %[[TMP_A:.*]] = load i8, ptr %[[A_ADDR]], align 1
// SHARED: %[[TMP_A_VEC:.*]] = bitcast i8 %[[TMP_A]] to <8 x i1>
// SHARED: %[[TMP_B:.*]] = load i8, ptr %[[B_ADDR]], align 1
// SHARED: %[[TMP_B_VEC:.*]] = bitcast i8 %[[TMP_B]] to <8 x i1>
// SHARED: %[[XOR:.*]] = xor <8 x i1> %[[TMP_A_VEC]], %[[TMP_B_VEC]]
// SHARED: %[[XOR_I8:.*]] = bitcast <8 x i1> %[[XOR]] to i8
// SHARED: store i8 %[[XOR_I8]], ptr %[[XOR_ADDR]], align 1
