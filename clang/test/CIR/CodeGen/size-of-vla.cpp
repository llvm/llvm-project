// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void vla_type_with_element_type_of_size_1() {
  unsigned long n = 10ul;
  unsigned long size = sizeof(bool[n]);
}

// CIR: %[[N_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["n", init]
// CIR: %[[SIZE_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["size", init]
// CIR: %[[CONST_10:.*]] = cir.const #cir.int<10> : !u64i
// CIR: cir.store {{.*}} %[[CONST_10]], %[[N_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_N:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: cir.store {{.*}} %[[TMP_N]], %[[SIZE_ADDR]] : !u64i, !cir.ptr<!u64i>

// LLVM: %[[N_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[SIZE_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: store i64 10, ptr %[[N_ADDR]], align 8
// LLVM: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// LLVM: store i64 %[[TMP_N]], ptr %[[SIZE_ADDR]], align 8

// OGCG: %[[N_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// OGCG: store i64 10, ptr %[[N_ADDR]], align 8
// OGCG: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// OGCG: store i64 %[[TMP_N]], ptr %[[SIZE_ADDR]], align 8

void vla_type_with_element_type_int() {
  unsigned long n = 10ul;
  unsigned long size = sizeof(int[n]);
}

// CIR: %[[N_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["n", init]
// CIR: %[[SIZE_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["size", init]
// CIR: %[[CONST_10:.*]] = cir.const #cir.int<10> : !u64i
// CIR: cir.store {{.*}} %[[CONST_10]], %[[N_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %3 = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !u64i
// CIR: %[[SIZE:.*]] = cir.binop(mul, %[[CONST_4]], %3) nuw : !u64i
// CIR: cir.store {{.*}} %[[SIZE]], %[[SIZE_ADDR]] : !u64i, !cir.ptr<!u64i>

// LLVM: %[[N_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[SIZE_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: store i64 10, ptr %[[N_ADDR]], align 8
// LLVM: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// LLVM: %[[SIZE:.*]] = mul nuw i64 4, %[[TMP_N]]
// LLVM: store i64 %[[SIZE]], ptr %[[SIZE_ADDR]], align 8

// OGCG: %[[N_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// OGCG: store i64 10, ptr %[[N_ADDR]], align 8
// OGCG: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// OGCG: %[[SIZE:.*]] = mul nuw i64 4, %[[TMP_N]]
// OGCG: store i64 %[[SIZE]], ptr %[[SIZE_ADDR]], align 8

void vla_expr_element_type_of_size_1() {
  unsigned long n = 10ul;
  bool arr[n];
  unsigned long size = sizeof(arr);
}

// CIR: %[[N_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["n", init]
// CIR: %[[SAVED_STACK_ADDR:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR: %[[CONST_10:.*]] = cir.const #cir.int<10> : !u64i
// CIR: cir.store {{.*}} %[[CONST_10]], %[[N_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_N:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[STACK_SAVE:.*]] = cir.stacksave : !cir.ptr<!u8i>
// CIR: cir.store {{.*}} %[[STACK_SAVE]], %[[SAVED_STACK_ADDR]] : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR: %[[ARR_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, %[[TMP_N]] : !u64i, ["arr"]
// CIR: %[[SIZE_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["size", init]
// CIR: cir.store {{.*}} %[[TMP_N]], %[[SIZE_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_SAVED_STACK:.*]] = cir.load {{.*}} %[[SAVED_STACK_ADDR]] : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CIR: cir.stackrestore %[[TMP_SAVED_STACK]] : !cir.ptr<!u8i>

// LLVM: %[[N_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[SAVED_STACK_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store i64 10, ptr %[[N_ADDR]], align 8
// LLVM: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// LLVM: %[[STACK_SAVE:.*]] = call ptr @llvm.stacksave.p0()
// LLVM: store ptr %[[STACK_SAVE]], ptr %[[SAVED_STACK_ADDR]], align 8
// LLVM: %[[ARR_ADDR:.*]] = alloca i8, i64 %[[TMP_N]], align 16
// LLVM: %[[SIZE_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: store i64 %[[TMP_N]], ptr %[[SIZE_ADDR]], align 8
// LLVM: %[[TMP_SAVED_STACK:.*]] = load ptr, ptr %[[SAVED_STACK_ADDR]], align 8
// LLVM: call void @llvm.stackrestore.p0(ptr %[[TMP_SAVED_STACK]])

// Note: VLA_EXPR0 below is emitted to capture debug info.

// OGCG: %[[N_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[SAVED_STACK_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[VLA_EXPR0:.*]] = alloca i64, align 8
// OGCG: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// OGCG: store i64 10, ptr %[[N_ADDR]], align 8
// OGCG: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// OGCG: %[[STACK_SAVE:.*]] = call ptr @llvm.stacksave.p0()
// OGCG: store ptr %[[STACK_SAVE]], ptr %[[SAVED_STACK_ADDR]], align 8
// OGCG: %[[ARR_ADDR:.*]] = alloca i8, i64 %[[TMP_N]], align 16
// OGCG: store i64 %[[TMP_N]], ptr %[[VLA_EXPR0]], align 8
// OGCG: store i64 %[[TMP_N]], ptr %[[SIZE_ADDR]], align 8
// OGCG: %[[TMP_SAVED_STACK:.*]] = load ptr, ptr %[[SAVED_STACK_ADDR]], align 8
// OGCG: call void @llvm.stackrestore.p0(ptr %[[TMP_SAVED_STACK]])

void vla_expr_element_type_int() {
  unsigned long n = 10ul;
  int arr[n];
  unsigned long size = sizeof(arr);
}

// CIR: %[[N_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["n", init]
// CIR: %[[SAVED_STACK_ADDR:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR: %[[CONST_10:.*]] = cir.const #cir.int<10> : !u64i
// CIR: cir.store {{.*}} %[[CONST_10]], %[[N_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_N:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!u64i>, !u64i
// CIR: %[[STACK_SAVE:.*]] = cir.stacksave : !cir.ptr<!u8i>
// CIR: cir.store {{.*}} %[[STACK_SAVE]], %[[SAVED_STACK_ADDR]] : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR: %[[ARR_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[TMP_N]] : !u64i, ["arr"]
// CIR: %[[SIZE_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["size", init]
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !u64i
// CIR: %[[SIZE:.*]] = cir.binop(mul, %[[CONST_4]], %[[TMP_N]]) nuw : !u64i
// CIR: cir.store {{.*}} %[[SIZE]], %[[SIZE_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_SAVED_STACK:.*]] = cir.load {{.*}} %[[SAVED_STACK_ADDR]] : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CIR: cir.stackrestore %[[TMP_SAVED_STACK]] : !cir.ptr<!u8i>

// LLVM: %[[N_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[SAVED_STACK_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store i64 10, ptr %[[N_ADDR]], align 8
// LLVM: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// LLVM: %[[STACK_SAVE:.*]] = call ptr @llvm.stacksave.p0()
// LLVM: store ptr %[[STACK_SAVE]], ptr %[[SAVED_STACK_ADDR]], align 8
// LLVM: %[[ARR_ADDR:.*]] = alloca i32, i64 %[[TMP_N]], align 16
// LLVM: %[[SIZE_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[SIZE:.*]] = mul nuw i64 4, %[[TMP_N]]
// LLVM: store i64 %[[SIZE]], ptr %[[SIZE_ADDR]], align 8
// LLVM: %[[TMP_SAVED_STACK:.*]] = load ptr, ptr %[[SAVED_STACK_ADDR]], align 8
// LLVM: call void @llvm.stackrestore.p0(ptr %[[TMP_SAVED_STACK]])

// Note: VLA_EXPR0 below is emitted to capture debug info.

// OGCG: %[[N_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[SAVED_STACK_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[VLA_EXPR0:.*]] = alloca i64, align 8
// OGCG: %[[SIZE_ADDR:.*]] = alloca i64, align 8
// OGCG: store i64 10, ptr %[[N_ADDR]], align 8
// OGCG: %[[TMP_N:.*]] = load i64, ptr %[[N_ADDR]], align 8
// OGCG: %[[STACK_SAVE:.*]] = call ptr @llvm.stacksave.p0()
// OGCG: store ptr %[[STACK_SAVE]], ptr %[[SAVED_STACK_ADDR]], align 8
// OGCG: %[[ARR_ADDR:.*]] = alloca i32, i64 %[[TMP_N]], align 16
// OGCG: store i64 %[[TMP_N]], ptr %[[VLA_EXPR0]], align 8
// OGCG: %[[SIZE:.*]] = mul nuw i64 4, %[[TMP_N]]
// OGCG: store i64 %[[SIZE]], ptr %[[SIZE_ADDR]], align 8
// OGCG: %[[TMP_SAVED_STACK:.*]] = load ptr, ptr %[[SAVED_STACK_ADDR]], align 8
// OGCG: call void @llvm.stackrestore.p0(ptr %[[TMP_SAVED_STACK]])
