// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR-DAG: cir.global "private" constant cir_private @__const.foo.bar = #cir.const_array<[#cir.fp<9.000000e+00> : !cir.double, #cir.fp<8.000000e+00> : !cir.double, #cir.fp<7.000000e+00> : !cir.double]> : !cir.array<!cir.double x 3>
typedef struct {
  int a;
  long b;
} T;

void buz(int x) {
  T arr[] = { {0, x}, {0, 0} };
}
// CIR: cir.func @buz
// CIR-NEXT: [[X_ALLOCA:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR-NEXT: [[ARR:%.*]] = cir.alloca !cir.array<!ty_T x 2>, !cir.ptr<!cir.array<!ty_T x 2>>, ["arr", init] {alignment = 16 : i64}
// CIR-NEXT: cir.store %arg0, [[X_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[ARR_INIT:%.*]] = cir.const #cir.zero : !cir.array<!ty_T x 2>
// CIR-NEXT: cir.store [[ARR_INIT]], [[ARR]] : !cir.array<!ty_T x 2>, !cir.ptr<!cir.array<!ty_T x 2>>
// CIR-NEXT: [[FI_EL:%.*]] = cir.cast(array_to_ptrdecay, [[ARR]] : !cir.ptr<!cir.array<!ty_T x 2>>), !cir.ptr<!ty_T>
// CIR-NEXT: [[A_STORAGE0:%.*]] = cir.get_member [[FI_EL]][0] {name = "a"} : !cir.ptr<!ty_T> -> !cir.ptr<!s32i>
// CIR-NEXT: [[B_STORAGE0:%.*]] = cir.get_member [[FI_EL]][1] {name = "b"} : !cir.ptr<!ty_T> -> !cir.ptr<!s64i>
// CIR-NEXT: [[X_VAL:%.*]] = cir.load [[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: [[X_CASTED:%.*]] = cir.cast(integral, [[X_VAL]] : !s32i), !s64i
// CIR-NEXT: cir.store [[X_CASTED]], [[B_STORAGE0]] : !s64i, !cir.ptr<!s64i>
// CIR-NEXT: [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: [[SE_EL:%.*]] = cir.ptr_stride([[FI_EL]] : !cir.ptr<!ty_T>, [[ONE]] : !s64i), !cir.ptr<!ty_T>
// CIR-NEXT: [[A_STORAGE1:%.*]] = cir.get_member [[SE_EL]][0] {name = "a"} : !cir.ptr<!ty_T> -> !cir.ptr<!s32i>
// CIR-NEXT: [[B_STORAGE1:%.*]] = cir.get_member [[SE_EL]][1] {name = "b"} : !cir.ptr<!ty_T> -> !cir.ptr<!s64i>
// CIR-NEXT: cir.return

void foo() {
  double bar[] = {9,8,7};
}
// CIR-LABEL: @foo
// CIR:  %[[DST:.*]] = cir.alloca !cir.array<!cir.double x 3>, !cir.ptr<!cir.array<!cir.double x 3>>, ["bar"]
// CIR:  %[[SRC:.*]] = cir.get_global @__const.foo.bar : !cir.ptr<!cir.array<!cir.double x 3>>
// CIR:  cir.copy %[[SRC]] to %[[DST]] : !cir.ptr<!cir.array<!cir.double x 3>>

void bar(int a, int b, int c) {
  int arr[] = {a,b,c};
}
// CIR: cir.func @bar
// CIR:      [[ARR:%.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["arr", init] {alignment = 4 : i64}
// CIR-NEXT: cir.store %arg0, [[A:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store %arg1, [[B:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store %arg2, [[C:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[FI_EL:%.*]] = cir.cast(array_to_ptrdecay, [[ARR]] : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>
// CIR-NEXT: [[LOAD_A:%.*]] = cir.load [[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store [[LOAD_A]], [[FI_EL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: [[SE_EL:%.*]] = cir.ptr_stride(%4 : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CIR-NEXT: [[LOAD_B:%.*]] = cir.load [[B]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store [[LOAD_B]], [[SE_EL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[TWO:%.*]] = cir.const #cir.int<2> : !s64i
// CIR-NEXT: [[TH_EL:%.*]] = cir.ptr_stride(%4 : !cir.ptr<!s32i>, [[TWO]] : !s64i), !cir.ptr<!s32i>
// CIR-NEXT: [[LOAD_C:%.*]] = cir.load [[C]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store [[LOAD_C]], [[TH_EL]] : !s32i, !cir.ptr<!s32i>

void zero_init(int x) {
  int arr[3] = {x};
}
// CIR:  cir.func @zero_init
// CIR:    [[VAR_ALLOC:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR:    %1 = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["arr", init] {alignment = 4 : i64}
// CIR:    [[TEMP:%.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init] {alignment = 8 : i64}
// CIR:    cir.store %arg0, [[VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[BEGIN:%.*]] = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>
// CIR:    [[VAR:%.*]] = cir.load [[VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.store [[VAR]], [[BEGIN]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:    [[ZERO_INIT_START:%.*]] = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CIR:    cir.store [[ZERO_INIT_START]], [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:    [[SIZE:%.*]] = cir.const #cir.int<3> : !s64i
// CIR:    [[END:%.*]] = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[SIZE]] : !s64i), !cir.ptr<!s32i>
// CIR:    cir.do {
// CIR:      [[CUR:%.*]] = cir.load [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:      [[FILLER:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:      cir.store [[FILLER]], [[CUR]] : !s32i, !cir.ptr<!s32i>
// CIR:      [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:      [[NEXT:%.*]] = cir.ptr_stride([[CUR]] : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CIR:      cir.store [[NEXT]], [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:      cir.yield
// CIR:    } while {
// CIR:      [[CUR:%.*]] = cir.load [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:      [[CMP:%.*]] = cir.cmp(ne, [[CUR]], [[END]]) : !cir.ptr<!s32i>, !cir.bool
// CIR:      cir.condition([[CMP]])
// CIR:    }
// CIR:    cir.return

void aggr_init() {
  int g = 5;
  int g_arr[5] = {1, 2, 3, g};
}
// CIR-LABEL:  cir.func no_proto @aggr_init
// CIR:    [[VAR_ALLOC:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["g", init] {alignment = 4 : i64}
// CIR:    %1 = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["g_arr", init] {alignment = 16 : i64}
// CIR:    [[TEMP:%.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init] {alignment = 8 : i64}
// CIR:    %3 = cir.const #cir.int<5> : !s32i
// CIR:    cir.store %3, [[VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[BEGIN:%.*]] = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CIR:    %5 = cir.const #cir.int<1> : !s32i
// CIR:    cir.store %5, [[BEGIN]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:    %7 = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CIR:    %8 = cir.const #cir.int<2> : !s32i
// CIR:    cir.store %8, %7 : !s32i, !cir.ptr<!s32i>
// CIR:    [[TWO:%.*]] = cir.const #cir.int<2> : !s64i
// CIR:    %10 = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[TWO]] : !s64i), !cir.ptr<!s32i>
// CIR:    %11 = cir.const #cir.int<3> : !s32i
// CIR:    cir.store %11, %10 : !s32i, !cir.ptr<!s32i>
// CIR:    [[THREE:%.*]] = cir.const #cir.int<3> : !s64i
// CIR:    %13 = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[THREE]] : !s64i), !cir.ptr<!s32i>
// CIR:    [[VAR:%.*]] = cir.load [[VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.store [[VAR]], %13 : !s32i, !cir.ptr<!s32i>
// CIR:    [[ONE_VAR:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:    %16 = cir.ptr_stride(%13 : !cir.ptr<!s32i>, [[ONE_VAR]] : !s64i), !cir.ptr<!s32i>
// CIR:    cir.store %16, [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:    [[SIZE:%.*]] = cir.const #cir.int<5> : !s64i
// CIR:    [[END:%.*]] = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[SIZE]] : !s64i), !cir.ptr<!s32i>
// CIR:    cir.do {
// CIR:      [[CUR:%.*]] = cir.load [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:      [[FILLER:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:      cir.store [[FILLER]], [[CUR]] : !s32i, !cir.ptr<!s32i>
// CIR:      [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:      [[NEXT:%.*]] = cir.ptr_stride([[CUR]] : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CIR:      cir.store [[NEXT]], [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:      cir.yield
// CIR:    } while {
// CIR:      [[CUR:%.*]] = cir.load [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:      [[CMP:%.*]] = cir.cmp(ne, [[CUR]], [[END]]) : !cir.ptr<!s32i>, !cir.bool
// CIR:      cir.condition([[CMP]])
// CIR:    }
// CIR:    cir.return
//
// LLVM-LABEL:  @aggr_init
// LLVM:   [[VAR_ALLOC:%.*]] = alloca i32, i64 1, align 4
// LLVM:   %2 = alloca [5 x i32], i64 1, align 16
// LLVM:   [[TEMP:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   store i32 5, ptr [[VAR_ALLOC]], align 4
// LLVM:   [[BEGIN:%.*]] = getelementptr i32, ptr %2, i32 0
// LLVM:   store i32 1, ptr [[BEGIN]], align 4
// LLVM:   [[ONE:%.*]] = getelementptr i32, ptr [[BEGIN]], i64 1
// LLVM:   store i32 2, ptr [[ONE]], align 4
// LLVM:   [[TWO:%.*]] = getelementptr i32, ptr [[BEGIN]], i64 2
// LLVM:   store i32 3, ptr [[TWO]], align 4
// LLVM:   [[THREE:%.*]] = getelementptr i32, ptr [[BEGIN]], i64 3
// LLVM:   [[VAR:%.*]] = load i32, ptr [[VAR_ALLOC]], align 4
// LLVM:   store i32 [[VAR]], ptr [[THREE]], align 4
// LLVM:   %9 = getelementptr i32, ptr [[THREE]], i64 1
// LLVM:   store ptr %9, ptr [[TEMP]], align 8
// LLVM:   [[END:%.*]] = getelementptr i32, ptr [[BEGIN]], i64 5
// LLVM:   br label %14
//
// LLVM: 11:                                               ; preds = %14
// LLVM:   [[CUR:%.*]] = load ptr, ptr [[TEMP]], align 8
// LLVM:   [[CMP:%.*]] = icmp ne ptr [[CUR]], [[END]]
// LLVM:   br i1 [[CMP]], label %14, label %17
//
// LLVM: 14:                                               ; preds = %11, %0
// LLVM:   [[CUR:%.*]] = load ptr, ptr [[TEMP]], align 8
// LLVM:   store i32 0, ptr [[CUR]], align 4
// LLVM:   [[NEXT:%.*]] = getelementptr i32, ptr [[CUR]], i64 1
// LLVM:   store ptr [[NEXT]], ptr [[TEMP]], align 8
// LLVM:   br label %11
//
// LLVM: 17:                                               ; preds = %11
// LLVM:   ret void
