// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,CIRONLY --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

unsigned _BitInt(200) w200 = 5;
// CIR-DAG: cir.global external @w200 = #cir.int<5> : !cir.int<u, 200, bitint> {alignment = 8 : i64}
// LLVM-DAG: @w200 = global i256 5, align 8
signed _BitInt(256) w256 = -1;
// CIR-DAG: cir.global external @w256 = #cir.int<-1> : !s256i_bitint {alignment = 8 : i64}
// LLVM-DAG: @w256 = global i256 -1, align 8

signed _BitInt(129) g_neg = -1;
// CIR-DAG: cir.global external @g_neg = #cir.int<-1> : !cir.int<s, 129, bitint> {alignment = 8 : i64}
// LLVM-DAG: @g_neg = global [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", align 8

struct HasWide129 {
  int i;
  signed _BitInt(129) bi;
};
struct HasWide129 g_struct = {7, -1};
// CIR-DAG: cir.global external @g_struct = #cir.const_record<{#cir.int<7> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<-1> : !cir.int<s, 129, bitint>}> : !rec_HasWide129 {alignment = 8 : i64}
// Note: CIR packs this, but classic codegen doesn't.  Layout is still
// identical, but there is a slight IR difference.
// LLVM-DAG: %struct.HasWide129 = type {{[<]?}}{ i32, [4 x i8], [24 x i8] }{{[>]?}}
// LLVM-DAG: @g_struct = global %struct.HasWide129 {{[<]?}}{ i32 7, [4 x i8] zeroinitializer, [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF" }{{[>]?}}, align 8

signed _BitInt(129) g_arr[2] = {-1, -2};
// CIR-DAG: cir.global external @g_arr = #cir.const_array<[#cir.int<-1> : !cir.int<s, 129, bitint>, #cir.int<-2> : !cir.int<s, 129, bitint>]> : !cir.array<!cir.int<s, 129, bitint> x 2> {alignment = 16 : i64}
// LLVM-DAG: @g_arr = global [2 x [24 x i8]] {{\[\[}}24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", [24 x i8] c"\FE\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF"], align 16

struct HasWide129Array {
  int i;
  signed _BitInt(129) bi[2];
};
struct HasWide129Array g_arr2 = {7, {-1, -2}};
// CIR-DAG: cir.global external @g_arr2 = #cir.const_record<{#cir.int<7> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.const_array<[#cir.int<-1> : !cir.int<s, 129, bitint>, #cir.int<-2> : !cir.int<s, 129, bitint>]> : !cir.array<!cir.int<s, 129, bitint> x 2>}> : !rec_HasWide129Array {alignment = 8 : i64}
// LLVM-DAG: %struct.HasWide129Array = type {{[<]?}}{ i32, [4 x i8], [2 x [24 x i8]] }{{[>]?}}
// LLVM-DAG: @g_arr2 = global %struct.HasWide129Array {{[<]?}}{ i32 7, [4 x i8] zeroinitializer, [2 x [24 x i8]] {{\[\[}}24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", [24 x i8] c"\FE\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF"] }{{[>]?}}, align 8

signed _BitInt(320) g320 = -1;
// CIR-DAG: cir.global external @g320 = #cir.int<-1> : !cir.int<s, 320, bitint> {alignment = 8 : i64}
// LLVM-DAG: @g320 = global [40 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", align 8

_BitInt(129) local_var(int a) {
  signed _BitInt(129) x = a;
  return x;
}
// CIR-LABEL: cir.func{{.*}}@local_var
// CIR-SAME: %[[RET_ARG:.*]]: !cir.ptr<!cir.int<s, 129, bitint>>
// CIR-SAME: %[[A_ARG:.*]]: !s32i
// CIR: %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR: %[[X:.*]] = cir.alloca "x" align(8) init : !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: cir.store %[[A_ARG]], %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[A_LOAD:.*]] = cir.load align(4) %[[A]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[CAST:.*]] = cir.cast integral %[[A_LOAD:.*]] : !s32i -> !cir.int<s, 129, bitint>
// CIR: cir.store align(8) %[[CAST]], %[[X]] : !cir.int<s, 129, bitint>, !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: %[[X_LOAD:.*]] = cir.load align(8) %[[X]] : !cir.ptr<!cir.int<s, 129, bitint>>, !cir.int<s, 129, bitint>
// CIR: cir.store %[[X_LOAD]], %[[RET_ARG]] : !cir.int<s, 129, bitint>, !cir.ptr<!cir.int<s, 129, bitint>>

// LLVM-LABEL: define{{.*}}@local_var
// LLVM-SAME: ptr{{.*}} %[[RET_ARG:.*]],
// LLVM-SAME: i32{{.*}} %[[ARG:.*]])
// LLVM: %[[A:.*]] = alloca i32, align 4
// LLVM: %[[X:.*]] = alloca [24 x i8], align 8
// LLVM: store i32 %[[ARG]], ptr %[[A]], align 4
// LLVM: %[[LOAD_A:.*]] = load i32, ptr %[[A]], align 4
// LLVM: %[[EXT_A:.*]] = sext i32 %[[LOAD_A]] to i129
// LLVM: %[[PAD_A:.*]] = sext i129 %[[EXT_A]] to i192
// LLVM: store i192 %[[PAD_A]], ptr %[[X]], align 8
// LLVM: %[[LOAD_X:.*]] = load i192, ptr %[[X]], align 8
// LLVM: %[[TRUNC_X:.*]] = trunc i192 %[[LOAD_X]] to i129
// LLVM: %[[EXT_X:.*]] = sext i129 %[[TRUNC_X]] to i192
// LLVM: store i192 %[[EXT_X]], ptr %[[RET_ARG]], align 8

_BitInt(129) local_arr(void) {
  signed _BitInt(129) arr[2] = {-1, -2};
  arr[1] = 3;

  return arr[0];
}

// CIR-LABEL: cir.func{{.*}}@local_arr
// CIR-SAME: %[[RET_ARG:.*]]: !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: %[[ARR:.*]] = cir.alloca "arr" align(16) init : !cir.ptr<!cir.array<!cir.int<s, 129, bitint> x 2>>
// CIR: %[[CONST_ARR:.*]] = cir.get_global @__const.local_arr.arr : !cir.ptr<!cir.array<!cir.int<s, 129, bitint> x 2>>
// CIR: cir.copy %[[CONST_ARR]] align(16) to %[[ARR]] align(16) : !cir.ptr<!cir.array<!cir.int<s, 129, bitint> x 2>>
// CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !cir.int<s, 129, bitint>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[GET_ELT_ONE:.*]] = cir.get_element %[[ARR]][%[[ONE]] : !s64i] : !cir.ptr<!cir.array<!cir.int<s, 129, bitint> x 2>> -> !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: cir.store align(8) %[[THREE]], %[[GET_ELT_ONE]] : !cir.int<s, 129, bitint>, !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CIR: %[[GET_ELT_ZERO:.*]] = cir.get_element %[[ARR]][%[[ZERO]] : !s64i] : !cir.ptr<!cir.array<!cir.int<s, 129, bitint> x 2>> -> !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: %[[LOAD_ELT_ZERO:.*]] = cir.load align(16) %[[GET_ELT_ZERO]] : !cir.ptr<!cir.int<s, 129, bitint>>, !cir.int<s, 129, bitint>
// CIR: cir.store %[[LOAD_ELT_ZERO]], %[[RET_ARG]] : !cir.int<s, 129, bitint>, !cir.ptr<!cir.int<s, 129, bitint>>

// LLVM-LABEL: define{{.*}}@local_arr
// LLVM-SAME: ptr{{.*}}%[[RET_ARG:.*]])
// LLVM: %[[ARR:.*]] = alloca [2 x [24 x i8]], align 16
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[ARR]], ptr align 16 @__const.local_arr.arr, i64 48, i1 false)
// LLVM: %[[GET_ELT_ONE:.*]] = getelementptr{{.*}} [2 x [24 x i8]], ptr %[[ARR]], i{{.*}} 0, i64 1
// LLVM: store i192 3, ptr %[[GET_ELT_ONE]], align 8
// LLVM: %[[GET_ELT_ZERO:.*]] = getelementptr{{.*}} [2 x [24 x i8]], ptr %[[ARR]], i{{.*}} 0, i64 0
// LLVM: %[[LOAD_ELT_ZERO:.*]] = load i192, ptr %[[GET_ELT_ZERO]], align 16
// LLVM: %[[TRUNC_ELT:.*]] = trunc i192 %[[LOAD_ELT_ZERO]] to i129
// LLVM: %[[EXT_ELT:.*]] = sext i129 %[[TRUNC_ELT]] to i192
// LLVM: store i192 %[[EXT_ELT]], ptr %[[RET_ARG]], align 8

_BitInt(129) take_param(signed _BitInt(129) x) {
return x;
}

// CIR-LABEL: cir.func{{.*}}@take_param
// CIR-SAME: %[[RET_ARG:[^:]*]]: !cir.ptr<!cir.int<s, 129, bitint>>
// CIR-SAME: %[[ARG:.*]]: !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: %[[LOAD_ARG:.*]] = cir.load %[[ARG]] : !cir.ptr<!cir.int<s, 129, bitint>>, !cir.int<s, 129, bitint>
// CIR: %[[ALLOCA_X:.*]] = cir.alloca "x" align(8) init : !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: cir.store %[[LOAD_ARG]], %[[ALLOCA_X]] : !cir.int<s, 129, bitint>, !cir.ptr<!cir.int<s, 129, bitint>>
// CIR: %[[LOAD_X:.*]] = cir.load align(8) %[[ALLOCA_X]] : !cir.ptr<!cir.int<s, 129, bitint>>, !cir.int<s, 129, bitint>
// CIR: cir.store %[[LOAD_X]], %[[RET_ARG]] : !cir.int<s, 129, bitint>, !cir.ptr<!cir.int<s, 129, bitint>>

// LLVM-LABEL: define{{.*}}@take_param
// LLVM-SAME: ptr{{.*}} %[[RET_ARG:.*]], 
// LLVM-SAME: ptr{{.*}}%[[ARG:.*]])
// OGCG: %[[X:.*]] = alloca [24 x i8], align 8
// LLVM: %[[LOAD_ARG:.*]] = load i192, ptr %[[ARG]], align 8
// LLVM: %[[TRUNC_ARG:.*]] = trunc i192 %[[LOAD_ARG]] to i129
// CIRONLY: %[[X:.*]] = alloca [24 x i8], align 8
// LLVM: %[[EXT_TRUNC_ARG:.*]] = sext i129 %[[TRUNC_ARG]] to i192
// LLVM: store i192 %[[EXT_TRUNC_ARG]], ptr %[[X]], align 8
// LLVM: %[[LOAD_X:.*]] = load i192, ptr %[[X]], align 8
// LLVM: %[[TRUNC_X:.*]] = trunc i192 %[[LOAD_X]] to i129
// LLVM: %[[EXT_X:.*]] = sext i129 %[[TRUNC_X]] to i192
// LLVM: store i192 %[[EXT_X]], ptr %[[RET_ARG]], align 8

signed _BitInt(129) ret_wide(void) { return 1; }

// CIR-LABEL: cir.func{{.*}}@ret_wide
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !cir.int<s, 129, bitint>

// LLVM-LABEL: define{{.*}}@ret_wide
// LLVM: store i192 1, ptr %{{.*}}, align 8

