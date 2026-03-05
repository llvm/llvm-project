// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR-DAG: cir.global "private" constant cir_private @__const.foo.bar = #cir.const_array<[#cir.fp<9.000000e+00> : !cir.double, #cir.fp<8.000000e+00> : !cir.double, #cir.fp<7.000000e+00> : !cir.double]> : !cir.array<!cir.double x 3>
typedef struct {
  int a;
  long b;
} T;

// Test array initialization with different elements.
typedef struct {
     long a0;
     int a1;
} Inner;
typedef struct {
     int b0;
     Inner b1[1];
} Outer;
Outer outers[2] = {
    {1, {0, 1} },
    {1, {0, 0} }
};
// CIR:  cir.global{{.*}} @outers =
// CIR-SAME: #cir.const_record<{
// CIR-SAME:   #cir.const_record<{
// CIR-SAME:     #cir.int<1> : !s32i,
// CIR-SAME:     #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 4>,
// CIR-SAME:     #cir.const_array<[
// CIR-SAME:       #cir.const_record<{#cir.int<0> : !s64i,
// CIR-SAME:                          #cir.int<1> : !s32i,
// CIR-SAME:                          #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 4>
// CIR-SAME:       }> : !rec_anon_struct
// CIR-SAME:     ]> : !cir.array<!rec_anon_struct x 1>
// CIR-SAME:   }> : !rec_anon_struct2,
// CIR-SAME:   #cir.const_record<{#cir.int<1> : !s32i,
// CIR-SAME:                      #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 4>,
// CIR-SAME:                      #cir.zero : !cir.array<!rec_Inner x 1>
// CIR-SAME:   }> : !rec_anon_struct1
// CIR-SAME: }> : !rec_anon_struct3
// LLVM: @outers = {{.*}}global
// LLVM-SAME: {
// LLVM-SAME:   { i32, [4 x i8], [1 x { i64, i32, [4 x i8] }] },
// LLVM-SAME:   { i32, [4 x i8], [1 x %struct.Inner] }
// LLVM-SAME: }
// LLVM-SAME: {
// LLVM-SAME:   { i32, [4 x i8], [1 x { i64, i32, [4 x i8] }] }
// LLVM-SAME:    { i32 1, [4 x i8] zeroinitializer, [1 x { i64, i32, [4 x i8] }] [{ i64, i32, [4 x i8] } { i64 0, i32 1, [4 x i8] zeroinitializer }] },
// LLVM-SAME:   { i32, [4 x i8], [1 x %struct.Inner] }
// LLVM-SAME:    { i32 1, [4 x i8] zeroinitializer, [1 x %struct.Inner] zeroinitializer }
// LLVM-SAME: }

void buz(int x) {
  T arr[] = { {x, x}, {0, 0} };
}
// CIR: cir.func {{.*}} @buz
// CIR-NEXT: [[X_ALLOCA:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR-NEXT: [[ARR:%.*]] = cir.alloca !cir.array<!rec_T x 2>, !cir.ptr<!cir.array<!rec_T x 2>>, ["arr", init] {alignment = 16 : i64}
// CIR-NEXT: cir.store{{.*}} %arg0, [[X_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: [[FI_EL:%.*]] = cir.get_element [[ARR]][[[ZERO]]] : (!cir.ptr<!cir.array<!rec_T x 2>>, !s32i) -> !cir.ptr<!rec_T>
// CIR-NEXT: [[A_STORAGE0:%.*]] = cir.get_member [[FI_EL]][0] {name = "a"} : !cir.ptr<!rec_T> -> !cir.ptr<!s32i>
// CIR-NEXT: [[XA_VAL:%.*]] = cir.load{{.*}} [[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store{{.*}} [[XA_VAL]], [[A_STORAGE0]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[B_STORAGE0:%.*]] = cir.get_member [[FI_EL]][1] {name = "b"} : !cir.ptr<!rec_T> -> !cir.ptr<!s64i>
// CIR-NEXT: [[XB_VAL:%.*]] = cir.load{{.*}} [[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: [[XB_CASTED:%.*]] = cir.cast integral [[XB_VAL]] : !s32i -> !s64i
// CIR-NEXT: cir.store{{.*}} [[XB_CASTED]], [[B_STORAGE0]] : !s64i, !cir.ptr<!s64i>
// CIR-NEXT: [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: [[SE_EL:%.*]] = cir.get_element [[ARR]][[[ONE]]] : (!cir.ptr<!cir.array<!rec_T x 2>>, !s64i) -> !cir.ptr<!rec_T>
// CIR-NEXT: [[A_STORAGE1:%.*]] = cir.get_member [[SE_EL]][0] {name = "a"} : !cir.ptr<!rec_T> -> !cir.ptr<!s32i>
// CIR-NEXT: [[A1_ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: cir.store{{.*}} [[A1_ZERO]], [[A_STORAGE1]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[B_STORAGE1:%.*]] = cir.get_member [[SE_EL]][1] {name = "b"} : !cir.ptr<!rec_T> -> !cir.ptr<!s64i>
// CIR-NEXT: [[B1_ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: [[B1_CASTED:%.*]] = cir.cast integral [[B1_ZERO]] : !s32i -> !s64i
// CIR-NEXT: cir.store{{.*}} [[B1_CASTED]], [[B_STORAGE1]] : !s64i, !cir.ptr<!s64i>
// CIR-NEXT: cir.return

void foo() {
  double bar[] = {9,8,7};
}
// CIR-LABEL: @foo
// CIR:  %[[DST:.*]] = cir.alloca !cir.array<!cir.double x 3>, !cir.ptr<!cir.array<!cir.double x 3>>, ["bar", init]
// CIR:  %[[SRC:.*]] = cir.get_global @__const.foo.bar : !cir.ptr<!cir.array<!cir.double x 3>>
// CIR:  cir.copy %[[SRC]] to %[[DST]] : !cir.ptr<!cir.array<!cir.double x 3>>

void bar(int a, int b, int c) {
  int arr[] = {a,b,c};
}
// CIR: cir.func {{.*}} @bar
// CIR:      [[ARR:%.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["arr", init] {alignment = 4 : i64}
// CIR-NEXT: cir.store{{.*}} %arg0, [[A:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store{{.*}} %arg1, [[B:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: cir.store{{.*}} %arg2, [[C:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT: [[ELEM0:%.*]] = cir.get_element [[ARR]][[[ZERO]]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!s32i>
// CIR-NEXT: [[LOAD_A:%.*]] = cir.load{{.*}} [[A]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store{{.*}} [[LOAD_A]], [[ELEM0]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: [[ELEM1:%.*]] = cir.get_element [[ARR]][[[ONE]]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s64i) -> !cir.ptr<!s32i>
// CIR-NEXT: [[LOAD_B:%.*]] = cir.load{{.*}} [[B]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store{{.*}} [[LOAD_B]], [[ELEM1]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: [[TWO:%.*]] = cir.const #cir.int<2> : !s64i
// CIR-NEXT: [[ELEM2:%.*]] = cir.get_element [[ARR]][[[TWO]]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s64i) -> !cir.ptr<!s32i>
// CIR-NEXT: [[LOAD_C:%.*]] = cir.load{{.*}} [[C]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.store{{.*}} [[LOAD_C]], [[ELEM2]] : !s32i, !cir.ptr<!s32i>

void zero_init(int x) {
  int arr[3] = {x};
}
// CIR:  cir.func {{.*}} @zero_init
// CIR:    [[VAR_ALLOC:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR:    [[ARR:%.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["arr", init] {alignment = 4 : i64}
// CIR:    [[TEMP:%.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init] {alignment = 8 : i64}
// CIR:    cir.store{{.*}} %arg0, [[VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:    [[BEGIN:%.*]] = cir.get_element [[ARR]][[[ZERO]]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!s32i>
// CIR:    [[VAR:%.*]] = cir.load{{.*}} [[VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.store{{.*}} [[VAR]], [[BEGIN]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:    [[ZERO_INIT_START:%.*]] = cir.get_element [[ARR]][[[ONE]]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s64i) -> !cir.ptr<!s32i>
// CIR:    cir.store{{.*}} [[ZERO_INIT_START]], [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:    [[SIZE:%.*]] = cir.const #cir.int<3> : !s64i
// CIR:    [[END:%.*]] = cir.get_element [[ARR]][[[SIZE]]] : (!cir.ptr<!cir.array<!s32i x 3>>, !s64i) -> !cir.ptr<!s32i>
// CIR:    cir.do {
// CIR:      [[CUR:%.*]] = cir.load{{.*}} [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:      [[FILLER:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:      cir.store{{.*}} [[FILLER]], [[CUR]] : !s32i, !cir.ptr<!s32i>
// CIR:      [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:      [[NEXT:%.*]] = cir.ptr_stride [[CUR]], [[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR:      cir.store{{.*}} [[NEXT]], [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:      cir.yield
// CIR:    } while {
// CIR:      [[CUR:%.*]] = cir.load{{.*}} [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:      [[CMP:%.*]] = cir.cmp(ne, [[CUR]], [[END]]) : !cir.ptr<!s32i>, !cir.bool
// CIR:      cir.condition([[CMP]])
// CIR:    }
// CIR:    cir.return

void aggr_init() {
  int g = 5;
  int g_arr[5] = {1, 2, 3, g};
}
// CIR-LABEL:  cir.func {{.*}} @aggr_init
// CIR:    [[VAR_ALLOC:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["g", init] {alignment = 4 : i64}
// CIR:    %1 = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["g_arr", init] {alignment = 16 : i64}
// CIR:    [[TEMP:%.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init] {alignment = 8 : i64}
// CIR:    %3 = cir.const #cir.int<5> : !s32i
// CIR:    cir.store{{.*}} %3, [[VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[OFFSET0:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:    [[BEGIN:%.*]] = cir.get_element %1[[[OFFSET0]]] : (!cir.ptr<!cir.array<!s32i x 5>>, !s32i) -> !cir.ptr<!s32i>
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:    cir.store{{.*}} [[ONE]], [[BEGIN]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[OFFSET1:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:    [[ELEM1:%.*]] = cir.get_element %1[[[OFFSET1]]] : (!cir.ptr<!cir.array<!s32i x 5>>, !s64i) -> !cir.ptr<!s32i>
// CIR:    [[TWO:%.*]] = cir.const #cir.int<2> : !s32i
// CIR:    cir.store{{.*}} [[TWO]], [[ELEM1]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[OFFSET2:%.*]] = cir.const #cir.int<2> : !s64i
// CIR:    [[ELEM2:%.*]] = cir.get_element %1[[[OFFSET2]]] : (!cir.ptr<!cir.array<!s32i x 5>>, !s64i) -> !cir.ptr<!s32i>
// CIR:    [[THREE:%.*]] = cir.const #cir.int<3> : !s32i
// CIR:    cir.store{{.*}} [[THREE]], [[ELEM2]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[OFFSET3:%.*]] = cir.const #cir.int<3> : !s64i
// CIR:    [[ELEM3:%.*]] = cir.get_element %1[[[OFFSET3]]] : (!cir.ptr<!cir.array<!s32i x 5>>, !s64i) -> !cir.ptr<!s32i>
// CIR:    [[VAR:%.*]] = cir.load{{.*}} [[VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.store{{.*}} [[VAR]], [[ELEM3]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[ONE_VAR:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:    [[OFFSET4:%.*]] = cir.binop(add, [[OFFSET3]], [[ONE_VAR]]) : !s64i
// CIR:    [[LAST:%.*]] = cir.get_element %1[[[OFFSET4]]] : (!cir.ptr<!cir.array<!s32i x 5>>, !s64i) -> !cir.ptr<!s32i>
// CIR:    cir.store{{.*}} [[LAST]], [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:    [[SIZE:%.*]] = cir.const #cir.int<5> : !s64i
// CIR:    [[END:%.*]] = cir.get_element %1[[[SIZE]]] : (!cir.ptr<!cir.array<!s32i x 5>>, !s64i) -> !cir.ptr<!s32i>
// CIR:    cir.do {
// CIR:      [[CUR:%.*]] = cir.load{{.*}} [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:      [[FILLER:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:      cir.store{{.*}} [[FILLER]], [[CUR]] : !s32i, !cir.ptr<!s32i>
// CIR:      [[ONE:%.*]] = cir.const #cir.int<1> : !s64i
// CIR:      [[NEXT:%.*]] = cir.ptr_stride [[CUR]], [[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR:      cir.store{{.*}} [[NEXT]], [[TEMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:      cir.yield
// CIR:    } while {
// CIR:      [[CUR:%.*]] = cir.load{{.*}} [[TEMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
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
// LLVM:   [[BEGIN:%.*]] = getelementptr [5 x i32], ptr %2, i32 0, i64 0
// LLVM:   store i32 1, ptr [[BEGIN]], align 4
// LLVM:   [[ONE:%.*]] = getelementptr [5 x i32], ptr %2, i32 0, i64 1
// LLVM:   store i32 2, ptr [[ONE]], align 4
// LLVM:   [[TWO:%.*]] = getelementptr [5 x i32], ptr %2, i32 0, i64 2
// LLVM:   store i32 3, ptr [[TWO]], align 4
// LLVM:   [[THREE:%.*]] = getelementptr [5 x i32], ptr %2, i32 0, i64 3
// LLVM:   [[VAR:%.*]] = load i32, ptr [[VAR_ALLOC]], align 4
// LLVM:   store i32 [[VAR]], ptr [[THREE]], align 4
// LLVM:   [[LAST:%.*]] = getelementptr [5 x i32], ptr %2, i32 0, i64 4
// LLVM:   store ptr [[LAST]], ptr [[TEMP]], align 8
// LLVM:   [[END:%.*]] = getelementptr [5 x i32], ptr %2, i32 0, i64 5
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
