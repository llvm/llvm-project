// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// CIR: cir.global "private" constant cir_private dso_local @[[ABC4:.*]] = #cir.const_array<"abc" : !cir.array<!s8i x 3>, trailing_zeros> : !cir.array<!s8i x 4>
// CIR: cir.global "private" constant cir_private dso_local @[[ABC15:.*]] = #cir.const_array<"abc" : !cir.array<!s8i x 3>, trailing_zeros> : !cir.array<!s8i x 15>
// CIR: cir.global "private" constant cir_private dso_local @[[HELLO:.*]] = #cir.const_array<"hello" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6>

// LLVM: @[[ABC4:.*]] = {{.*}}constant [4 x i8] c"abc\00", align 1
// LLVM: @[[ABC15:.*]] = {{.*}}constant [15 x i8] c"abc\00\00\00\00\00\00
// LLVM: @[[HELLO:.*]] = {{.*}}constant [6 x i8] c"hello\00", align 1

// CIR-LABEL: cir.func{{.*}} @_Z2fni(
// CHECK-LABEL: define{{.*}} void @_Z2fni
void fn(int n) {
  // CIR: %[[N_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
  // CIR: %[[N_LOAD:.*]] = cir.load {{.*}} %[[N_ALLOCA]] : !cir.ptr<!s32i>
  // CIR: %[[N_U64_CAST:.*]] = cir.cast integral %[[N_LOAD]] : !s32i -> !u64i
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !u64i
  // CIR: %[[N_LT_THREE:.*]] = cir.cmp lt %[[N_U64_CAST]], %[[THREE]] : !u64i
  // CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[N_IN_BYTES:.*]], %[[OVERFLOW:.*]] = cir.mul.overflow %[[N_U64_CAST]], %[[FOUR]] : !u64i -> !u64i
  // CIR: %[[LT_THREE_OR_OVRFL:.*]] = cir.or %[[N_LT_THREE]], %[[OVERFLOW]] : !cir.bool
  // 64 bit all 1s
  // CIR: %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
  // CIR: %[[ADJ_SIZE:.*]] = cir.select if %[[LT_THREE_OR_OVRFL]] then %[[ALL_ONES]] else %[[N_IN_BYTES]] : (!cir.bool, !u64i, !u64i) -> !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[ADJ_SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_TO_INTS:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}}%[[ONE]], %[[ALLOC_TO_INTS]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[ALLOC_TO_INTS]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}}%[[TWO]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[NEXT_ELT2:.*]] = cir.ptr_stride %[[NEXT_ELT]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}}%[[THREE]], %[[NEXT_ELT2]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[NEXT_ELT3:.*]] = cir.ptr_stride %[[NEXT_ELT2]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWELVE:.*]] = cir.const #cir.int<12> : !u64i
  // CIR: %[[REST_SIZE:.*]] = cir.sub %[[ADJ_SIZE]], %[[TWELVE]] : !u64i
  // CIR: %[[REST_ALLOC_AS_VOID:.*]] = cir.cast bitcast %[[NEXT_ELT3]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[REST_SIZE]] bytes at %[[REST_ALLOC_AS_VOID]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

  // LLVM: %[[N_ALLOCA:.*]] = alloca i32
  // LLVM: %[[N_LOAD:.*]] = load i32, ptr %[[N_ALLOCA]]
  // LLVM: %[[N_U64_CAST:.*]] = sext i32 %[[N_LOAD]] to i64
  // LLVM: %[[N_LT_THREE:.*]] = icmp ult i64 %[[N_U64_CAST]], 3
  // LLVM: %[[MUL_RES:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_U64_CAST]], i64 4)
  // LLVM-DAG: %[[N_IN_BYTES:.*]] = extractvalue { i64, i1 } %[[MUL_RES]], 0
  // LLVM-DAG: %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_RES]], 1
  // LLVM-DAG: %[[N_LT3_OR_OF:.*]] = or i1 %[[N_LT_THREE]], %[[OVERFLOW]]
  // LLVM-DAG: %[[ADJ_SIZE:.*]] = select i1 %[[N_LT3_OR_OF]], i64 -1, i64 %[[N_IN_BYTES]]
  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef %[[ADJ_SIZE]])
  // LLVM: store i32 1, ptr %[[ALLOC]]
  // LLVM: %[[ELT1:.*]] = getelementptr{{.*}}i32, ptr %[[ALLOC]], i64 1
  // LLVM: store i32 2, ptr %[[ELT1]]
  // LLVM: %[[ELT2:.*]] = getelementptr{{.*}}i32, ptr %[[ELT1]], i64 1
  // LLVM: store i32 3, ptr %[[ELT2]]
  // LLVM: %[[ELT3:.*]] = getelementptr{{.*}}i32, ptr %[[ELT2]], i64 1
  // LLVM: %[[REST_SIZE:.*]] = sub i64 %[[ADJ_SIZE]], 12
  // call void @llvm.memset.p0.i64(ptr{{.*}} %[[ELT3]], i8 0, i64 %[[REST_SIZE:.*]], i1 false)
  new int[n] { 1, 2, 3 };
}

// CIR-LABEL: cir.func {{.*}}@_Z8fn_pareni(
// LLVM-LABEL: define{{.*}} void @_Z8fn_pareni
void fn_paren(int n) {
  // CIR: %[[N_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
  // CIR: %[[N_LOAD:.*]] = cir.load {{.*}} %[[N_ALLOCA]] : !cir.ptr<!s32i>
  // CIR: %[[N_U64_CAST:.*]] = cir.cast integral %[[N_LOAD]] : !s32i -> !u64i
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !u64i
  // CIR: %[[N_LT_THREE:.*]] = cir.cmp lt %[[N_U64_CAST]], %[[THREE]] : !u64i
  // CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[N_IN_BYTES:.*]], %[[OVERFLOW:.*]] = cir.mul.overflow %[[N_U64_CAST]], %[[FOUR]] : !u64i -> !u64i
  // CIR: %[[LT_THREE_OR_OVRFL:.*]] = cir.or %[[N_LT_THREE]], %[[OVERFLOW]] : !cir.bool
  // 64 bit all 1s
  // CIR: %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
  // CIR: %[[ADJ_SIZE:.*]] = cir.select if %[[LT_THREE_OR_OVRFL]] then %[[ALL_ONES]] else %[[N_IN_BYTES]] : (!cir.bool, !u64i, !u64i) -> !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[ADJ_SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_TO_INTS:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}}%[[ONE]], %[[ALLOC_TO_INTS]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[ALLOC_TO_INTS]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}}%[[TWO]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[NEXT_ELT2:.*]] = cir.ptr_stride %[[NEXT_ELT]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}}%[[THREE]], %[[NEXT_ELT2]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[NEXT_ELT3:.*]] = cir.ptr_stride %[[NEXT_ELT2]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWELVE:.*]] = cir.const #cir.int<12> : !u64i
  // CIR: %[[REST_SIZE:.*]] = cir.sub %[[ADJ_SIZE]], %[[TWELVE]] : !u64i
  // CIR: %[[REST_ALLOC_AS_VOID:.*]] = cir.cast bitcast %[[NEXT_ELT3]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[REST_SIZE]] bytes at %[[REST_ALLOC_AS_VOID]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

  // LLVM: %[[N_ALLOCA:.*]] = alloca i32
  // LLVM: %[[N_LOAD:.*]] = load i32, ptr %[[N_ALLOCA]]
  // LLVM: %[[N_U64_CAST:.*]] = sext i32 %[[N_LOAD]] to i64
  // LLVM: %[[N_LT_THREE:.*]] = icmp ult i64 %[[N_U64_CAST]], 3
  // LLVM: %[[MUL_RES:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_U64_CAST]], i64 4)
  // LLVM-DAG: %[[N_IN_BYTES:.*]] = extractvalue { i64, i1 } %[[MUL_RES]], 0
  // LLVM-DAG: %[[OVERFLOW:.*]] = extractvalue { i64, i1 } %[[MUL_RES]], 1
  // LLVM-DAG: %[[N_LT3_OR_OF:.*]] = or i1 %[[N_LT_THREE]], %[[OVERFLOW]]
  // LLVM-DAG: %[[ADJ_SIZE:.*]] = select i1 %[[N_LT3_OR_OF]], i64 -1, i64 %[[N_IN_BYTES]]
  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef %[[ADJ_SIZE]])
  // LLVM: store i32 1, ptr %[[ALLOC]]
  // LLVM: %[[ELT1:.*]] = getelementptr{{.*}}i32, ptr %[[ALLOC]], i64 1
  // LLVM: store i32 2, ptr %[[ELT1]]
  // LLVM: %[[ELT2:.*]] = getelementptr{{.*}}i32, ptr %[[ELT1]], i64 1
  // LLVM: store i32 3, ptr %[[ELT2]]
  // LLVM: %[[ELT3:.*]] = getelementptr{{.*}}i32, ptr %[[ELT2]], i64 1
  // LLVM: %[[REST_SIZE:.*]] = sub i64 %[[ADJ_SIZE]], 12
  // call void @llvm.memset.p0.i64(ptr{{.*}} %[[ELT3]], i8 0, i64 %[[REST_SIZE:.*]], i1 false)
  new int[n](1, 2, 3);
}

// CIR-LABEL: cir.func {{.*}}@_Z11const_exactv()
// LLVM-LABEL: define{{.*}} void @_Z11const_exactv
void const_exact() {
  // CIR: %[[ALLOC_SIZE:.*]] = cir.const #cir.int<12> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_TO_INTS:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}}%[[ONE]], %[[ALLOC_TO_INTS]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_TO_INTS]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}}%[[TWO]], %[[ELT1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}}%[[THREE]], %[[ELT2]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT3:.*]] = cir.ptr_stride %[[ELT2]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 12)
  // LLVM: store i32 1, ptr %[[ALLOC]]
  // LLVM: %[[ELT1:.*]] = getelementptr{{.*}}i32, ptr %[[ALLOC]], i64 1
  // LLVM: store i32 2, ptr %[[ELT1]]
  // LLVM: %[[ELT2:.*]] = getelementptr{{.*}}i32, ptr %[[ELT1]], i64 1
  // LLVM: store i32 3, ptr %[[ELT2]]
  // LLVM: %[[REST:.*]] = getelementptr{{.*}}i32, ptr %[[ELT2]], i64 1
  // LLVM: ret void
  new int[3] { 1, 2, 3 };
}

// CIR-LABEL: cir.func {{.*}}@_Z17const_exact_parenv()
// LLVM-LABEL: define{{.*}} void @_Z17const_exact_parenv
void const_exact_paren() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<12> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}} %[[ONE]], %[[ALLOC_CAST]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_CAST]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}} %[[TWO]], %[[ELT1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}} %[[THREE]], %[[ELT2]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT3:.*]] = cir.ptr_stride %[[ELT2]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 12)
  // LLVM: store i32 1, ptr %[[ALLOC]]
  // LLVM: %[[ELT1:.*]] = getelementptr{{.*}}i32, ptr %[[ALLOC]], i64 1
  // LLVM: store i32 2, ptr %[[ELT1]]
  // LLVM: %[[ELT2:.*]] = getelementptr{{.*}}i32, ptr %[[ELT1]], i64 1
  // LLVM: store i32 3, ptr %[[ELT2]]
  // LLVM: %[[REST:.*]] = getelementptr{{.*}}i32, ptr %[[ELT2]], i64 1
  // LLVM: ret void
  new int[3](1, 2, 3);
}

// CIR-LABEL: cir.func {{.*}}@_Z16const_sufficientv()
// LLVM-LABEL: define{{.*}} void @_Z16const_sufficientv
void const_sufficient() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<16> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_INTS:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}} %[[ONE]], %[[ALLOC_INTS]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_INTS]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}} %[[TWO]], %[[ELT1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}} %[[THREE]], %[[ELT2]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT3:.*]] = cir.ptr_stride %[[ELT2]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[INIT_SIZE:.*]] = cir.const #cir.int<12> : !u64i
  // CIR: %[[REST_SIZE:.*]] = cir.sub %[[SIZE]], %[[INIT_SIZE]] : !u64i
  // CIR: %[[REST_PTR_DECAY:.*]] = cir.cast bitcast %[[ELT3]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[REST_SIZE]] bytes at %[[REST_PTR_DECAY]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 16)
  // LLVM: store i32 1, ptr %[[ALLOC]]
  // LLVM: %[[ELT1:.*]] = getelementptr{{.*}}i32, ptr %[[ALLOC]], i64 1
  // LLVM: store i32 2, ptr %[[ELT1]]
  // LLVM: %[[ELT2:.*]] = getelementptr{{.*}}i32, ptr %[[ELT1]], i64 1
  // LLVM: store i32 3, ptr %[[ELT2]]
  // LLVM: %[[REST:.*]] = getelementptr{{.*}}i32, ptr %[[ELT2]], i64 1
  // LLVM: call void @llvm.memset.p0.i64(ptr{{.*}} %[[REST]], i8 0, i64 4, i1 false)
  // LLVM: ret void
  new int[4] { 1, 2, 3 };
}

// CIR-LABEL: cir.func {{.*}}@_Z22const_sufficient_parenv()
// LLVM-LABEL: define{{.*}} void @_Z22const_sufficient_parenv
void const_sufficient_paren() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<16> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_INTS:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}} %[[ONE]], %[[ALLOC_INTS]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_INTS]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}} %[[TWO]], %[[ELT1]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}} %[[THREE]], %[[ELT2]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT3:.*]] = cir.ptr_stride %[[ELT2]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
  // CIR: %[[INIT_SIZE:.*]] = cir.const #cir.int<12> : !u64i
  // CIR: %[[REST_SIZE:.*]] = cir.sub %[[SIZE]], %[[INIT_SIZE]] : !u64i
  // CIR: %[[REST_PTR_DECAY:.*]] = cir.cast bitcast %[[ELT3]] : !cir.ptr<!s32i> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[REST_SIZE]] bytes at %[[REST_PTR_DECAY]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 16)
  // LLVM: store i32 1, ptr %[[ALLOC]]
  // LLVM: %[[ELT1:.*]] = getelementptr{{.*}}i32, ptr %[[ALLOC]], i64 1
  // LLVM: store i32 2, ptr %[[ELT1]]
  // LLVM: %[[ELT2:.*]] = getelementptr{{.*}}i32, ptr %[[ELT1]], i64 1
  // LLVM: store i32 3, ptr %[[ELT2]]
  // LLVM: %[[REST:.*]] = getelementptr{{.*}}i32, ptr %[[ELT2]], i64 1
  // LLVM: call void @llvm.memset.p0.i64(ptr{{.*}} %[[REST]], i8 0, i64 4, i1 false)
  // LLVM: ret void
  new int[4](1, 2, 3);
  // CHECKCXX20: ret void
}

// TODO(cir): This still falls-through to the end of emitNewArrayInitializer as
// 'unsupported initializer' and requires a full loop with cleanups/etc. So this
// is likely not ready to be implemented yet.
// CHECK-LABEL: define{{.*}} void @_Z22check_array_value_initv
//void check_array_value_init() {
//  struct S;
//  new (int S::*[3][4][5]) ();
//
//  // CHECK: call noalias noundef nonnull ptr @_Zna{{.}}(i{{32 noundef 240|64 noundef 480}})
//  // CHECK: getelementptr inbounds i{{32|64}}, ptr {{.*}}, i{{32|64}} 60
//
//  // CHECK: phi
//  // CHECK: store i{{32|64}} -1,
//  // CHECK: getelementptr inbounds i{{32|64}}, ptr {{.*}}, i{{32|64}} 1
//  // CHECK: icmp eq
//  // CHECK: br i1
//}

// CIR-LABEL: cir.func {{.*}}@_Z15string_nonconsti(
// LLVM-LABEL: define{{.*}} void @_Z15string_nonconsti
void string_nonconst(int n) {
  // CIR: %[[N_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
  // CIR: %[[N_LOAD:.*]] = cir.load {{.*}} %[[N_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[N_CAST:.*]] = cir.cast integral %[[N_LOAD]] : !s32i -> !u64i
  // CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[N_LT_4:.*]] = cir.cmp lt %[[N_CAST]], %[[FOUR]] : !u64i
  // CIR: %[[NEG_ONE:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
  // CIR: %[[SIZE:.*]] = cir.select if %[[N_LT_4]] then %[[NEG_ONE]] else %[[N_CAST]] : (!cir.bool, !u64i, !u64i) -> !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC4]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[CONST_STR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[AFTER_COPY:.*]] = cir.ptr_stride %[[ALLOC_CAST]], %[[CONST_STR_SIZE]] : (!cir.ptr<!s8i>, !u64i) -> !cir.ptr<!s8i>
  // CIR: %[[CONST_STR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[SIZE_LEFT:.*]] = cir.sub %[[SIZE]], %[[CONST_STR_SIZE]] : !u64i
  // CIR: %[[AFTER_COPY_CAST:.*]] = cir.cast bitcast %[[AFTER_COPY]] : !cir.ptr<!s8i> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[SIZE_LEFT]] bytes at %[[AFTER_COPY_CAST]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

  // LLVM: %[[ARG_ALLOCA:.*]] = alloca i32
  // LLVM: %[[ARG_LOAD:.*]] = load i32, ptr %[[ARG_ALLOCA]]
  // LLVM: %[[ARG_CAST:.*]] = sext i32 %[[ARG_LOAD]] to i64
  // LLVM: %[[N_LT_4:.*]] = icmp ult i64 %[[ARG_CAST]], 4
  // LLVM: %[[SIZE:.*]] = select i1 %[[N_LT_4]], i64 -1, i64 %[[ARG_CAST]]
  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef %[[SIZE]])
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC4]], i64 4, i1 false)
  // LLVM: %[[AFTER_COPY:.*]] = getelementptr {{.*}}i8, ptr %[[ALLOC]]
  // LLVM: %[[SIZE_LEFT:.*]] = sub i64 %[[SIZE]], 4
  // LLVM: call void @llvm.memset.p0.i64(ptr{{.*}} %[[AFTER_COPY]], i8 0, i64 %[[SIZE_LEFT]], i1 false)
  new char[n] { "abc" };
}

// CIR-LABEL: cir.func no_inline dso_local @_Z21string_nonconst_pareni(
// LLVM-LABEL: define{{.*}} void @_Z21string_nonconst_pareni
void string_nonconst_paren(int n) {
  // CIR: %[[N_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
  // CIR: %[[N_LOAD:.*]] = cir.load {{.*}} %[[N_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[N_CAST:.*]] = cir.cast integral %[[N_LOAD]] : !s32i -> !u64i
  // CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[N_LT_4:.*]] = cir.cmp lt %[[N_CAST]], %[[FOUR]] : !u64i
  // CIR: %[[NEG_ONE:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
  // CIR: %[[SIZE:.*]] = cir.select if %[[N_LT_4]] then %[[NEG_ONE]] else %[[N_CAST]] : (!cir.bool, !u64i, !u64i) -> !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC4]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[CONST_STR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[AFTER_COPY:.*]] = cir.ptr_stride %[[ALLOC_CAST]], %[[CONST_STR_SIZE]] : (!cir.ptr<!s8i>, !u64i) -> !cir.ptr<!s8i>
  // CIR: %[[CONST_STR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[SIZE_LEFT:.*]] = cir.sub %[[SIZE]], %[[CONST_STR_SIZE]] : !u64i
  // CIR: %[[AFTER_COPY_CAST:.*]] = cir.cast bitcast %[[AFTER_COPY]] : !cir.ptr<!s8i> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[SIZE_LEFT]] bytes at %[[AFTER_COPY_CAST]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

  // LLVM: %[[ARG_ALLOCA:.*]] = alloca i32
  // LLVM: %[[ARG_LOAD:.*]] = load i32, ptr %[[ARG_ALLOCA]]
  // LLVM: %[[ARG_CAST:.*]] = sext i32 %[[ARG_LOAD]] to i64
  // LLVM: %[[N_LT_4:.*]] = icmp ult i64 %[[ARG_CAST]], 4
  // LLVM: %[[SIZE:.*]] = select i1 %[[N_LT_4]], i64 -1, i64 %[[ARG_CAST]]
  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef %[[SIZE]])
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC4]], i64 4, i1 false)
  // LLVM: %[[AFTER_COPY:.*]] = getelementptr {{.*}}i8, ptr %[[ALLOC]]
  // LLVM: %[[SIZE_LEFT:.*]] = sub i64 %[[SIZE]], 4
  // LLVM: call void @llvm.memset.p0.i64(ptr{{.*}} %[[AFTER_COPY]], i8 0, i64 %[[SIZE_LEFT]], i1 false)
  new char[n]("abc");
}

// CIR-LABEL: cir.func {{.*}}@_Z33string_nonconst_paren_extra_pareni(
// LLVM-LABEL: define{{.*}} void @_Z33string_nonconst_paren_extra_pareni
void string_nonconst_paren_extra_paren(int n) {
  // CIR: %[[N_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
  // CIR: %[[N_LOAD:.*]] = cir.load {{.*}} %[[N_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[N_CAST:.*]] = cir.cast integral %[[N_LOAD]] : !s32i -> !u64i
  // CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[N_LT_4:.*]] = cir.cmp lt %[[N_CAST]], %[[FOUR]] : !u64i
  // CIR: %[[NEG_ONE:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
  // CIR: %[[SIZE:.*]] = cir.select if %[[N_LT_4]] then %[[NEG_ONE]] else %[[N_CAST]] : (!cir.bool, !u64i, !u64i) -> !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC4]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[CONST_STR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[AFTER_COPY:.*]] = cir.ptr_stride %[[ALLOC_CAST]], %[[CONST_STR_SIZE]] : (!cir.ptr<!s8i>, !u64i) -> !cir.ptr<!s8i>
  // CIR: %[[CONST_STR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[SIZE_LEFT:.*]] = cir.sub %[[SIZE]], %[[CONST_STR_SIZE]] : !u64i
  // CIR: %[[AFTER_COPY_CAST:.*]] = cir.cast bitcast %[[AFTER_COPY]] : !cir.ptr<!s8i> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[SIZE_LEFT]] bytes at %[[AFTER_COPY_CAST]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

  // LLVM: %[[ARG_ALLOCA:.*]] = alloca i32
  // LLVM: %[[ARG_LOAD:.*]] = load i32, ptr %[[ARG_ALLOCA]]
  // LLVM: %[[ARG_CAST:.*]] = sext i32 %[[ARG_LOAD]] to i64
  // LLVM: %[[N_LT_4:.*]] = icmp ult i64 %[[ARG_CAST]], 4
  // LLVM: %[[SIZE:.*]] = select i1 %[[N_LT_4]], i64 -1, i64 %[[ARG_CAST]]
  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef %[[SIZE]])
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC4]], i64 4, i1 false)
  // LLVM: %[[AFTER_COPY:.*]] = getelementptr {{.*}}i8, ptr %[[ALLOC]]
  // LLVM: %[[SIZE_LEFT:.*]] = sub i64 %[[SIZE]], 4
  // LLVM: call void @llvm.memset.p0.i64(ptr{{.*}} %[[AFTER_COPY]], i8 0, i64 %[[SIZE_LEFT]], i1 false)
  new char[n](("abc"));
}

// CIR-LABEL: cir.func {{.*}}@_Z12string_exactv()
// LLVM-LABEL: define{{.*}} void @_Z12string_exactv
void string_exact() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC4]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 4)
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC4]], i64 4, i1 false)
  new char[4] { "abc" };
}

// CIR-LABEL: cir.func {{.*}}@_Z18string_exact_parenv()
// LLVM-LABEL: define{{.*}} void @_Z18string_exact_parenv
void string_exact_paren() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC4]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 4)
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC4]], i64 4, i1 false)
  new char[4]("abc");
}

// CIR-LABEL: cir.func {{.*}}@_Z28string_exact_paren_extensionv()
// LLVM-LABEL: define{{.*}} void @_Z28string_exact_paren_extensionv
void string_exact_paren_extension() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC4]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 4>>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 4)
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC4]], i64 4, i1 false)
  new char[4](__extension__ "abc");
}

// CIR-LABEL: cir.func {{.*}}@_Z17string_sufficientv()
// LLVM-LABEL: define{{.*}} void @_Z17string_sufficientv
void string_sufficient() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<15> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 15>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC15]] : !cir.ptr<!cir.array<!s8i x 15>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 15>>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 15)
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC15]], i64 15, i1 false)
  new char[15] { "abc" };
}

// CIR-LABEL: cir.func {{.*}}@_Z23string_sufficient_parenv()
// LLVM-LABEL: define{{.*}} void @_Z23string_sufficient_parenv
void string_sufficient_paren() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<15> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_AS_STRING:.*]] = cir.cast bitcast %[[ALLOC_CAST]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 15>>
  // CIR: %[[GET_STR:.*]] = cir.get_global @[[ABC15]] : !cir.ptr<!cir.array<!s8i x 15>>
  // CIR: cir.copy %[[GET_STR]] to %[[ALLOC_AS_STRING]] : !cir.ptr<!cir.array<!s8i x 15>>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 15)
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[ABC15]], i64 15, i1 false)
  new char[15]("abc");
}

// CIR-LABEL: cir.func {{.*}}@_Z10aggr_exactv()
// LLVM-LABEL: define{{.*}} void @_Z10aggr_exactv
void aggr_exact() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<16> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!rec_Aggr>
  // CIR: %[[GET_A:.*]] = cir.get_member %[[ALLOC_CAST]][0] {name = "a"} : !cir.ptr<!rec_Aggr> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}} %[[ONE]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_B:.*]] = cir.get_member %[[ALLOC_CAST]][1] {name = "b"} : !cir.ptr<!rec_Aggr> -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}} %[[TWO]], %[[GET_B]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_CAST]], %[[ONE]] : (!cir.ptr<!rec_Aggr>, !s32i) -> !cir.ptr<!rec_Aggr>
  // CIR: %[[GET_A:.*]] = cir.get_member %[[ELT1]][0] {name = "a"} : !cir.ptr<!rec_Aggr> -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}} %[[THREE]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_B:.*]] = cir.get_member %[[ELT1]][1] {name = "b"} : !cir.ptr<!rec_Aggr> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!rec_Aggr>, !s32i) -> !cir.ptr<!rec_Aggr>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 16)
  // LLVM: %[[GET_A:.*]] = getelementptr {{.*}}%struct.Aggr, ptr %[[ALLOC]], i32 0, i32 0
  // LLVM: store i32 1, ptr %[[GET_A]]
  // LLVM: %[[GET_B:.*]] = getelementptr {{.*}}%struct.Aggr, ptr %[[ALLOC]], i32 0, i32 1
  // LLVM: store i32 2, ptr %[[GET_B]]
  // LLVM: %[[ELT1:.*]] = getelementptr {{.*}}%struct.Aggr, ptr %[[ALLOC]], i64 1
  // LLVM: %[[GET_A:.*]] = getelementptr {{.*}}%struct.Aggr, ptr %[[ELT1]], i32 0, i32 0
  // LLVM: store i32 3, ptr %[[GET_A]]
  // LLVM: %[[GET_B:.*]] = getelementptr {{.*}}%struct.Aggr, ptr %[[ELT1]], i32 0, i32 1
  // LLVM: store i32 0, ptr %[[GET_B]]
  // LLVM: %[[REST:.*]] = getelementptr {{.*}}%struct.Aggr, ptr %[[ELT1]], i64 1
  // LLVM: ret
  struct Aggr { int a, b; };
  new Aggr[2] { 1, 2, 3 };
}

// CIR-LABEL: cir.func {{.*}}@_Z15aggr_sufficienti(
// LLVM-LABEL: define{{.*}} void @_Z15aggr_sufficienti
void aggr_sufficient(int n) {
  // CIR: %[[ARG:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
  // CIR: %[[GET_N:.*]] = cir.load {{.*}}%[[ARG:.*]] : !cir.ptr<!s32i>, !s32i
  // CIR: %[[N_CAST:.*]] = cir.cast integral %[[GET_N]] : !s32i -> !u64i
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !u64i
  // CIR: %[[N_LT_2:.*]] = cir.cmp lt %[[N_CAST]], %[[TWO]] : !u64i
  // CIR: %[[AGG_SIZE:.*]] = cir.const #cir.int<8> : !u64i
  // CIR: %[[N_BYTES:.*]], %[[MUL_OF:.*]] = cir.mul.overflow %[[N_CAST]], %[[AGG_SIZE]] : !u64i -> !u64i
  // CIR: %[[N_LT_OR_OF:.*]] = cir.or %[[N_LT_2]], %[[MUL_OF]] : !cir.bool
  // CIR: %[[NEG_ONE:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
  // CIR: %[[SIZE:.*]] = cir.select if %[[N_LT_OR_OF]] then %[[NEG_ONE]] else %[[N_BYTES]] : (!cir.bool, !u64i, !u64i) -> !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!rec_Aggr2E0>
  // CIR: %[[GET_A:.*]] = cir.get_member %[[ALLOC_CAST]][0] {name = "a"} : !cir.ptr<!rec_Aggr2E0> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}} %[[ONE]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_B:.*]] = cir.get_member %[[ALLOC_CAST]][1] {name = "b"} : !cir.ptr<!rec_Aggr2E0> -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}} %[[TWO]], %[[GET_B]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_CAST]], %[[ONE]] : (!cir.ptr<!rec_Aggr2E0>, !s32i) -> !cir.ptr<!rec_Aggr2E0>
  // CIR: %[[GET_A:.*]] = cir.get_member %[[ELT1]][0] {name = "a"} : !cir.ptr<!rec_Aggr2E0> -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}} %[[THREE]], %[[GET_A]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_B:.*]] = cir.get_member %[[ELT1]][1] {name = "b"} : !cir.ptr<!rec_Aggr2E0> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!rec_Aggr2E0>, !s32i) -> !cir.ptr<!rec_Aggr2E0>
  // CIR: %[[TWO_ELTS_SIZE:.*]] = cir.const #cir.int<16> : !u64i
  // CIR: %[[REST_SIZE:.*]] = cir.sub %[[SIZE]], %[[TWO_ELTS_SIZE]] : !u64i
  // CIR: %[[REST_DECAY:.*]] = cir.cast bitcast %[[ELT2]] : !cir.ptr<!rec_Aggr2E0> -> !cir.ptr<!void>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
  // CIR: cir.libc.memset %[[REST_SIZE]] bytes at %[[REST_DECAY]] to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i

  // LLVM: %[[ARG:.*]] = alloca i32
  // LLVM: %[[GET_N:.*]] = load i32, ptr %[[ARG]]
  // LLVM: %[[N_CAST:.*]] = sext i32 %[[GET_N]] to i64
  // LLVM: %[[N_LT_2:.*]] = icmp ult i64 %[[N_CAST]], 2
  // LLVM: %[[MUL_RES:.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %[[N_CAST]], i64 8)
  // LLVM-DAG: %[[N_BYTES:.*]] = extractvalue { i64, i1 } %[[MUL_RES]], 0
  // LLVM-DAG: %[[MUL_OF:.*]] = extractvalue { i64, i1 } %[[MUL_RES]], 1
  // LLVM-DAG: %[[N_LT_OR_OF:.*]] = or i1 %[[N_LT_2]], %[[MUL_OF]]
  // LLVM-DAG: %[[SIZE:.*]] = select i1 %[[N_LT_OR_OF]], i64 -1, i64 %[[N_BYTES]]
  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef %[[SIZE]])
  // LLVM: %[[GET_A:.*]] = getelementptr {{.*}}%struct.Aggr.0, ptr %[[ALLOC]], i32 0, i32 0
  // LLVM: store i32 1, ptr %[[GET_A]]
  // LLVM: %[[GET_B:.*]] = getelementptr {{.*}}%struct.Aggr.0, ptr %[[ALLOC]], i32 0, i32 1
  // LLVM: store i32 2, ptr %[[GET_B]]
  // LLVM: %[[ELT1:.*]] = getelementptr {{.*}}%struct.Aggr.0, ptr %[[ALLOC]], i64 1
  // LLVM: %[[GET_A:.*]] = getelementptr {{.*}}%struct.Aggr.0, ptr %[[ELT1]], i32 0, i32 0
  // LLVM: store i32 3, ptr %[[GET_A]]
  // LLVM: %[[GET_B:.*]] = getelementptr {{.*}}%struct.Aggr.0, ptr %[[ELT1]], i32 0, i32 1
  // LLVM: store i32 0, ptr %[[GET_B]]
  // LLVM: %[[ELT2:.*]] = getelementptr {{.*}}%struct.Aggr.0, ptr %[[ELT1]], i64 1
  // LLVM: %[[REST_SIZE:.*]] = sub i64 %[[SIZE]], 16
  // LLVM: call void @llvm.memset.p0.i64(ptr{{.*}} %[[ELT2]], i8 0, i64 %[[REST_SIZE]], i1 false)
  struct Aggr { int a, b; };
  new Aggr[n] { 1, 2, 3 };
}

// CIR-LABEL: cir.func {{.*}}@_Z14constexpr_testv()
// LLVM-LABEL: define{{.*}} void @_Z14constexpr_testv
void constexpr_test() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<4> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_INTS:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store {{.*}} %[[ZERO]], %[[ALLOC_INTS]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_INTS]], %[[ONE]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 4)
  // LLVM: store i32 0, ptr %[[ALLOC]]
  // LLVM: %[[ELT1:.*]] = getelementptr{{.*}}i32, ptr %[[ALLOC]], i64 1

  new int[0+1]{0};
}

// CIR-LABEL: cir.func {{.*}}@_Z13unknown_boundv()
// LLVM-LABEL: define{{.*}} void @_Z13unknown_boundv
void unknown_bound() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<24> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_AGG:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!rec_Aggr2E1>
  // CIR: %[[GET_X:.*]] = cir.get_member %[[ALLOC_AGG]][0] {name = "x"} : !cir.ptr<!rec_Aggr2E1> -> !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.store {{.*}} %[[ONE]], %[[GET_X]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_Y:.*]] = cir.get_member %[[ALLOC_AGG]][1] {name = "y"} : !cir.ptr<!rec_Aggr2E1> -> !cir.ptr<!s32i>
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.store {{.*}} %[[TWO]], %[[GET_Y]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_Z:.*]] = cir.get_member %[[ALLOC_AGG]][2] {name = "z"} : !cir.ptr<!rec_Aggr2E1> -> !cir.ptr<!s32i>
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.store {{.*}} %[[THREE]], %[[GET_Z]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT1:.*]] = cir.ptr_stride %[[ALLOC_AGG]], %[[ONE]] : (!cir.ptr<!rec_Aggr2E1>, !s32i) -> !cir.ptr<!rec_Aggr2E1>
  // CIR: %[[GET_X:.*]] = cir.get_member %[[ELT1]][0] {name = "x"} : !cir.ptr<!rec_Aggr2E1> -> !cir.ptr<!s32i>
  // CIR: %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: cir.store {{.*}} %[[FOUR]], %[[GET_X]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_Y:.*]] = cir.get_member %[[ELT1]][1] {name = "y"} : !cir.ptr<!rec_Aggr2E1> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store {{.*}} %[[ZERO]], %[[GET_Y]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[GET_Z:.*]] = cir.get_member %[[ELT1]][2] {name = "z"} : !cir.ptr<!rec_Aggr2E1> -> !cir.ptr<!s32i>
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: cir.store {{.*}} %[[ZERO]], %[[GET_Z]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[ELT2:.*]] = cir.ptr_stride %[[ELT1]], %[[ONE]] : (!cir.ptr<!rec_Aggr2E1>, !s32i) -> !cir.ptr<!rec_Aggr2E1>

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 24)
  // LLVM: %[[GET_X:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ALLOC]], i32 0, i32 0
  // LLVM: store i32 1, ptr %[[GET_X]]
  // LLVM: %[[GET_Y:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ALLOC]], i32 0, i32 1
  // LLVM: store i32 2, ptr %[[GET_Y]]
  // LLVM: %[[GET_Z:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ALLOC]], i32 0, i32 2
  // LLVM: store i32 3, ptr %[[GET_Z]]
  // LLVM: %[[ELT1:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ALLOC]], i64 1
  // LLVM: %[[GET_X:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ELT1]], i32 0, i32 0
  // LLVM: store i32 4, ptr %[[GET_X]]
  // LLVM: %[[GET_Y:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ELT1]], i32 0, i32 1
  // LLVM: store i32 0, ptr %[[GET_Y]]
  // LLVM: %[[GET_Z:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ELT1]], i32 0, i32 2
  // LLVM: store i32 0, ptr %[[GET_Z]]
  // LLVM: %[[ELT2:.*]] = getelementptr {{.*}}%struct.Aggr.1, ptr %[[ELT1]], i64 1
  struct Aggr { int x, y, z; };
  new Aggr[]{1, 2, 3, 4};
}

// CIR-LABEL: cir.func {{.*}}@_Z20unknown_bound_stringv()
// LLVM-LABEL: define{{.*}} void @_Z20unknown_bound_stringv
void unknown_bound_string() {
  // CIR: %[[SIZE:.*]] = cir.const #cir.int<6> : !u64i
  // CIR: %[[ALLOC:.*]] = cir.call @_Znam(%[[SIZE]]) {allocsize = array<i32: 0>, builtin} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull, llvm.noundef})
  // CIR: %[[ALLOC_CHAR:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CIR: %[[ALLOC_STR:.*]] = cir.cast bitcast %[[ALLOC_CHAR]] : !cir.ptr<!s8i> -> !cir.ptr<!cir.array<!s8i x 6>>
  // CIR: %[[GET_HELLO:.*]] = cir.get_global @[[HELLO]] : !cir.ptr<!cir.array<!s8i x 6>>
  // CIR: cir.copy %[[GET_HELLO]] to %[[ALLOC_STR]] : !cir.ptr<!cir.array<!s8i x 6>>
  // CIR: cir.return

  // LLVM: %[[ALLOC:.*]] = call{{.*}}nonnull ptr @_Znam(i64 noundef 6)
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr{{.*}} %[[ALLOC]], ptr {{.*}}@[[HELLO]], i64 6, i1 false)
  new char[]{"hello"};
}
