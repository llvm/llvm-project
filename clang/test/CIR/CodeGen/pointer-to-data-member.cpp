// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Point {
  int x;
  int y;
  int z;
};

int Point::*pt_member = &Point::z;
// CIR-BEFORE: cir.global external @pt_member = #cir.data_member<2> : !cir.data_member<!s32i in !rec_Point>
// CIR-AFTER: cir.global external @pt_member = #cir.int<8> : !s64i
// LLVM: @pt_member = global i64 8
// OGCG: @pt_member = global i64 8

auto test1() -> int Point::* {
  return &Point::y;
}

int Point::*pt_member_nested_region = test1();

// CIR-BEFORE: cir.global external @pt_member_nested_region = ctor : !cir.data_member<!s32i in !rec_Point> {
// CIR-BEFORE:   %[[MEMBER_PTR_ADDR:.*]] = cir.get_global @pt_member_nested_region : !cir.ptr<!cir.data_member<!s32i in !rec_Point>>
// CIR-BEFORE:   %[[MEMBER_PTR:.*]] = cir.call @_Z5test1v() : () -> !cir.data_member<!s32i in !rec_Point>
// CIR-BEFORE:   cir.store{{.*}} %[[MEMBER_PTR]], %[[MEMBER_PTR_ADDR]] : !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>
// CIR-BEFORE: }

// CIR-AFTER: cir.global external @pt_member_nested_region = #cir.int<-1> : !s64i
// CIR-AFTER: cir.func internal private @__cxx_global_var_init()
// CIR-AFTER:   %[[MEMBER_PTR_ADDR:.*]] = cir.get_global @pt_member_nested_region : !cir.ptr<!s64i>
// CIR-AFTER:   %[[MEMBER_PTR:.*]] = cir.call @_Z5test1v() : () -> !s64i
// CIR-AFTER:   cir.store align(8) %[[MEMBER_PTR]], %[[MEMBER_PTR_ADDR]] : !s64i, !cir.ptr<!s64i>

// LLVM: @pt_member_nested_region = global i64 -1, align 8
// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   %[[MEMBER_PTR:.*]] = call i64 @_Z5test1v()
// LLVM:   store i64 %[[MEMBER_PTR]], ptr @pt_member_nested_region, align 8

// OGCG: @pt_member_nested_region = global i64 -1, align 8
// OGCG emits __cxx_global_var_init between test1() and test2(). See checks below.

// Checks for test1()

// CIR-BEFORE: cir.func {{.*}} @_Z5test1v() -> !cir.data_member<!s32i in !rec_Point> {
// CIR-BEFORE:   %[[RETVAL:.*]] = cir.alloca !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>, ["__retval"]
// CIR-BEFORE:   %[[MEMBER:.*]] = cir.const #cir.data_member<1> : !cir.data_member<!s32i in !rec_Point>
// CIR-BEFORE:   cir.store %[[MEMBER]], %[[RETVAL]] : !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>
// CIR-BEFORE:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!cir.data_member<!s32i in !rec_Point>>, !cir.data_member<!s32i in !rec_Point>
// CIR-BEFORE:   cir.return %[[RET]] : !cir.data_member<!s32i in !rec_Point>

// CIR-AFTER: cir.func {{.*}} @_Z5test1v() -> !s64i {
// CIR-AFTER:   %[[RETVAL:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["__retval"]
// CIR-AFTER:   %[[OFFSET:.*]] = cir.const #cir.int<4> : !s64i
// CIR-AFTER:   cir.store %[[OFFSET]], %[[RETVAL]] : !s64i, !cir.ptr<!s64i>
// CIR-AFTER:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   cir.return %[[RET]] : !s64i

// LLVM: define {{.*}} i64 @_Z5test1v()
// LLVM:   %[[RETVAL:.*]] = alloca i64
// LLVM:   store i64 4, ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// LLVM:   ret i64 %[[RET]]

// OGCG: define {{.*}} i64 @_Z5test1v()
// OGCG:   ret i64 4

// OGCG: define internal void @__cxx_global_var_init()
// OGCG:   %[[MEMBER_PTR:.*]] = call i64 @_Z5test1v()
// OGCG:   store i64 %[[MEMBER_PTR]], ptr @pt_member_nested_region


int test2(const Point &pt, int Point::*member) {
  return pt.*member;
}

// CIR-BEFORE:       cir.func {{.*}} @_Z5test2RK5PointMS_i(
// CIR-BEFORE-SAME:        %[[PT_ARG:.*]]: !cir.ptr<!rec_Point>
// CIR-BEFORE-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Point>
// CIR-BEFORE:         %[[PT_ADDR:.*]] = cir.alloca {{.*}} ["pt", init, const]
// CIR-BEFORE:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR-BEFORE:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR-BEFORE:         cir.store %[[PT_ARG]], %[[PT_ADDR]]
// CIR-BEFORE:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR-BEFORE:         %[[PT:.*]] = cir.load %[[PT_ADDR]]
// CIR-BEFORE:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR-BEFORE:         %[[RT_MEMBER:.*]] = cir.get_runtime_member %[[PT]][%[[MEMBER]] : !cir.data_member<!s32i in !rec_Point>] : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CIR-BEFORE:         %[[VAL:.*]] = cir.load{{.*}} %[[RT_MEMBER]]
// CIR-BEFORE:         cir.store %[[VAL]], %[[RETVAL_ADDR]]
// CIR-BEFORE:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR-BEFORE:         cir.return %[[RET]]

// CIR-AFTER:      cir.func {{.*}} @_Z5test2RK5PointMS_i(
// CIR-AFTER-SAME:        %[[PT_ARG:.*]]: !cir.ptr<!rec_Point>
// CIR-AFTER-SAME:        %[[MEMBER_ARG:.*]]: !s64i
// CIR-AFTER:        %[[PT_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Point>, !cir.ptr<!cir.ptr<!rec_Point>>, ["pt", init, const]
// CIR-AFTER:        %[[MEMBER_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["member", init]
// CIR-AFTER:        %[[RETVAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-AFTER:        cir.store %[[PT_ARG]], %[[PT_ADDR]] : !cir.ptr<!rec_Point>, !cir.ptr<!cir.ptr<!rec_Point>>
// CIR-AFTER:        cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]] : !s64i, !cir.ptr<!s64i>
// CIR-AFTER:        %[[PT:.*]] = cir.load %[[PT_ADDR]] : !cir.ptr<!cir.ptr<!rec_Point>>, !cir.ptr<!rec_Point>
// CIR-AFTER:        %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]] : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:        %[[BYTE_PTR:.*]] = cir.cast bitcast %[[PT]] : !cir.ptr<!rec_Point> -> !cir.ptr<!s8i>
// CIR-AFTER:        %[[BYTE_PTR_STRIDE:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[MEMBER]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>
// CIR-AFTER:        %[[VAL_ADDR:.*]] = cir.cast bitcast %[[BYTE_PTR_STRIDE]] : !cir.ptr<!s8i> -> !cir.ptr<!s32i>
// CIR-AFTER:        %[[VAL:.*]] = cir.load{{.*}} %[[VAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-AFTER:        cir.store %[[VAL]], %[[RETVAL_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-AFTER:        %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-AFTER:        cir.return %[[RET]] : !s32i

// LLVM: define {{.*}} i32 @_Z5test2RK5PointMS_i(ptr %[[PT_ARG:.*]], i64 %[[MEMBER_ARG:.*]])
// LLVM:   %[[PT_ADDR:.*]] = alloca ptr
// LLVM:   %[[MEMBER_ADDR:.*]] = alloca i64
// LLVM:   %[[RETVAL_ADDR:.*]] = alloca i32
// LLVM:   store ptr %[[PT_ARG]], ptr %[[PT_ADDR]]
// LLVM:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// LLVM:   %[[PT:.*]] = load ptr, ptr %[[PT_ADDR]]
// LLVM:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// LLVM:   %[[RT_MEMBER:.*]] = getelementptr i8, ptr %[[PT]], i64 %[[MEMBER]]
// LLVM:   %[[VAL:.*]] = load i32, ptr %[[RT_MEMBER]]
// LLVM:   store i32 %[[VAL]], ptr %[[RETVAL_ADDR]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL_ADDR]]
// LLVM:   ret i32 %[[RET]]

// OGCG: define {{.*}} i32 @_Z5test2RK5PointMS_i(ptr {{.*}} %[[PT_ARG:.*]], i64 %[[MEMBER_ARG:.*]])
// OGCG:   %[[PT_ADDR:.*]] = alloca ptr
// OGCG:   %[[MEMBER_ADDR:.*]] = alloca i64
// OGCG:   store ptr %[[PT_ARG]], ptr %[[PT_ADDR]]
// OGCG:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// OGCG:   %[[PT:.*]] = load ptr, ptr %[[PT_ADDR]]
// OGCG:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// OGCG:   %[[RT_MEMBER:.*]] = getelementptr inbounds i8, ptr %[[PT]], i64 %[[MEMBER]]
// OGCG:   %[[RET:.*]] = load i32, ptr %[[RT_MEMBER]]
// OGCG:   ret i32 %[[RET]]

int test3(const Point *pt, int Point::*member) {
  return pt->*member;
}

// CIR-BEFORE:       cir.func {{.*}} @_Z5test3PK5PointMS_i(
// CIR-BEFORE-SAME:        %[[PT_ARG:.*]]: !cir.ptr<!rec_Point>
// CIR-BEFORE-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Point>
// CIR-BEFORE:         %[[PT_ADDR:.*]] = cir.alloca {{.*}} ["pt", init]
// CIR-BEFORE:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR-BEFORE:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR-BEFORE:         cir.store %[[PT_ARG]], %[[PT_ADDR]]
// CIR-BEFORE:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR-BEFORE:         %[[PT:.*]] = cir.load{{.*}} %[[PT_ADDR]]
// CIR-BEFORE:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR-BEFORE:         %[[RT_MEMBER:.*]] = cir.get_runtime_member %[[PT]][%[[MEMBER]] : !cir.data_member<!s32i in !rec_Point>] : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CIR-BEFORE:         %[[VAL:.*]] = cir.load{{.*}} %[[RT_MEMBER]]
// CIR-BEFORE:         cir.store %[[VAL]], %[[RETVAL_ADDR]]
// CIR-BEFORE:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR-BEFORE:         cir.return %[[RET]]

// CIR-AFTER:      cir.func {{.*}} @_Z5test3PK5PointMS_i(
// CIR-AFTER-SAME:        %[[PT_ARG:.*]]: !cir.ptr<!rec_Point>
// CIR-AFTER-SAME:        %[[MEMBER_ARG:.*]]: !s64i
// CIR-AFTER:        %[[PT_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Point>, !cir.ptr<!cir.ptr<!rec_Point>>, ["pt", init]
// CIR-AFTER:        %[[MEMBER_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["member", init]
// CIR-AFTER:        %[[RETVAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-AFTER:        cir.store %[[PT_ARG]], %[[PT_ADDR]] : !cir.ptr<!rec_Point>, !cir.ptr<!cir.ptr<!rec_Point>>
// CIR-AFTER:        cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]] : !s64i, !cir.ptr<!s64i>
// CIR-AFTER:        %[[PT:.*]] = cir.load{{.*}} %[[PT_ADDR]] : !cir.ptr<!cir.ptr<!rec_Point>>, !cir.ptr<!rec_Point>
// CIR-AFTER:        %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]] : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:        %[[BYTE_PTR:.*]] = cir.cast bitcast %[[PT]] : !cir.ptr<!rec_Point> -> !cir.ptr<!s8i>
// CIR-AFTER:        %[[BYTE_PTR_STRIDE:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[MEMBER]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>
// CIR-AFTER:        %[[VAL_ADDR:.*]] = cir.cast bitcast %[[BYTE_PTR_STRIDE]] : !cir.ptr<!s8i> -> !cir.ptr<!s32i>
// CIR-AFTER:        %[[VAL:.*]] = cir.load{{.*}} %[[VAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-AFTER:        cir.store %[[VAL]], %[[RETVAL_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-AFTER:        %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-AFTER:        cir.return %[[RET]] : !s32i

// LLVM: define {{.*}} i32 @_Z5test3PK5PointMS_i(ptr %[[PT_ARG:.*]], i64 %[[MEMBER_ARG:.*]])
// LLVM:   %[[PT_ADDR:.*]] = alloca ptr
// LLVM:   %[[MEMBER_ADDR:.*]] = alloca i64
// LLVM:   %[[RETVAL_ADDR:.*]] = alloca i32
// LLVM:   store ptr %[[PT_ARG]], ptr %[[PT_ADDR]]
// LLVM:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// LLVM:   %[[PT:.*]] = load ptr, ptr %[[PT_ADDR]]
// LLVM:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// LLVM:   %[[RT_MEMBER:.*]] = getelementptr i8, ptr %[[PT]], i64 %[[MEMBER]]
// LLVM:   %[[VAL:.*]] = load i32, ptr %[[RT_MEMBER]]
// LLVM:   store i32 %[[VAL]], ptr %[[RETVAL_ADDR]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL_ADDR]]
// LLVM:   ret i32 %[[RET]]

// OGCG: define {{.*}} i32 @_Z5test3PK5PointMS_i(ptr {{.*}} %[[PT_ARG:.*]], i64 %[[MEMBER_ARG:.*]])
// OGCG:   %[[PT_ADDR:.*]] = alloca ptr
// OGCG:   %[[MEMBER_ADDR:.*]] = alloca i64
// OGCG:   store ptr %[[PT_ARG]], ptr %[[PT_ADDR]]
// OGCG:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// OGCG:   %[[PT:.*]] = load ptr, ptr %[[PT_ADDR]]
// OGCG:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// OGCG:   %[[RT_MEMBER:.*]] = getelementptr inbounds i8, ptr %[[PT]], i64 %[[MEMBER]]
// OGCG:   %[[RET:.*]] = load i32, ptr %[[RT_MEMBER]]
// OGCG:   ret i32 %[[RET]]

struct Incomplete;

auto test4(int Incomplete::*member) -> int Incomplete::* {
  return member;
}

// CIR-BEFORE:       cir.func {{.*}} @_Z5test4M10Incompletei(
// CIR-BEFORE-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Incomplete>
// CIR-BEFORE:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR-BEFORE:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR-BEFORE:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR-BEFORE:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR-BEFORE:         cir.store %[[MEMBER]], %[[RETVAL_ADDR]]
// CIR-BEFORE:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR-BEFORE:         cir.return %[[RET]]

// CIR-AFTER:      cir.func {{.*}} @_Z5test4M10Incompletei(
// CIR-AFTER-SAME:       %[[MEMBER_ARG:.*]]: !s64i
// CIR-AFTER:        %[[MEMBER_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["member", init]
// CIR-AFTER:        %[[RETVAL_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["__retval"]
// CIR-AFTER:        cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]] : !s64i, !cir.ptr<!s64i>
// CIR-AFTER:        %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]] : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:        cir.store %[[MEMBER]], %[[RETVAL_ADDR]] : !s64i, !cir.ptr<!s64i>
// CIR-AFTER:        %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]] : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:        cir.return %[[RET]] : !s64i

// LLVM: define {{.*}} i64 @_Z5test4M10Incompletei(i64 %[[MEMBER_ARG:.*]])
// LLVM:   %[[MEMBER_ADDR:.*]] = alloca i64
// LLVM:   %[[RETVAL_ADDR:.*]] = alloca i64
// LLVM:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// LLVM:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// LLVM:   store i64 %[[MEMBER]], ptr %[[RETVAL_ADDR]]
// LLVM:   %[[RET:.*]] = load i64, ptr %[[RETVAL_ADDR]]
// LLVM:   ret i64 %[[RET]]

// OGCG: define {{.*}} i64 @_Z5test4M10Incompletei(i64 %[[MEMBER_ARG:.*]])
// OGCG:   %[[MEMBER_ADDR:.*]] = alloca i64
// OGCG:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// OGCG:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// OGCG:   ret i64 %[[MEMBER]]

int test5(Incomplete *ic, int Incomplete::*member) {
  return ic->*member;
}

// CIR-BEFORE:       cir.func {{.*}} @_Z5test5P10IncompleteMS_i(
// CIR-BEFORE-SAME:        %[[IC_ARG:.*]]: !cir.ptr<!rec_Incomplete>
// CIR-BEFORE-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Incomplete>
// CIR-BEFORE:         %[[IC_ADDR:.*]] = cir.alloca {{.*}} ["ic", init]
// CIR-BEFORE:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR-BEFORE:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR-BEFORE:         cir.store %[[IC_ARG]], %[[IC_ADDR]]
// CIR-BEFORE:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR-BEFORE:         %[[IC:.*]] = cir.load{{.*}} %[[IC_ADDR]]
// CIR-BEFORE:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR-BEFORE:         %[[RT_MEMBER:.*]] = cir.get_runtime_member %[[IC]][%[[MEMBER]] : !cir.data_member<!s32i in !rec_Incomplete>] : !cir.ptr<!rec_Incomplete> -> !cir.ptr<!s32i>
// CIR-BEFORE:         %[[VAL:.*]] = cir.load{{.*}} %[[RT_MEMBER]]
// CIR-BEFORE:         cir.store %[[VAL]], %[[RETVAL_ADDR]]
// CIR-BEFORE:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR-BEFORE:         cir.return %[[RET]]

// CIR-AFTER:      cir.func {{.*}} @_Z5test5P10IncompleteMS_i(
// CIR-AFTER-SAME:       %[[IC_ARG:.*]]: !cir.ptr<!rec_Incomplete>
// CIR-AFTER-SAME:       %[[MEMBER_ARG:.*]]: !s64i
// CIR-AFTER:        %[[IC_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Incomplete>, !cir.ptr<!cir.ptr<!rec_Incomplete>>, ["ic", init]
// CIR-AFTER:        %[[MEMBER_ADDR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["member", init]
// CIR-AFTER:        %[[RETVAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-AFTER:        cir.store %[[IC_ARG]], %[[IC_ADDR]] : !cir.ptr<!rec_Incomplete>, !cir.ptr<!cir.ptr<!rec_Incomplete>>
// CIR-AFTER:        cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]] : !s64i, !cir.ptr<!s64i>
// CIR-AFTER:        %[[IC:.*]] = cir.load{{.*}} %[[IC_ADDR]] : !cir.ptr<!cir.ptr<!rec_Incomplete>>, !cir.ptr<!rec_Incomplete>
// CIR-AFTER:        %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]] : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:        %[[BYTE_PTR:.*]] = cir.cast bitcast %[[IC]] : !cir.ptr<!rec_Incomplete> -> !cir.ptr<!s8i>
// CIR-AFTER:        %[[BYTE_PTR_STRIDE:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[MEMBER]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>
// CIR-AFTER:        %[[VAL_ADDR:.*]] = cir.cast bitcast %[[BYTE_PTR_STRIDE]] : !cir.ptr<!s8i> -> !cir.ptr<!s32i>
// CIR-AFTER:        %[[VAL:.*]] = cir.load{{.*}} %[[VAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-AFTER:        cir.store %[[VAL]], %[[RETVAL_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR-AFTER:        %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-AFTER:        cir.return %[[RET]] : !s32i

// LLVM: define {{.*}} i32 @_Z5test5P10IncompleteMS_i(ptr %[[IC_ARG:.*]], i64 %[[MEMBER_ARG:.*]])
// LLVM:   %[[IC_ADDR:.*]] = alloca ptr
// LLVM:   %[[MEMBER_ADDR:.*]] = alloca i64
// LLVM:   %[[RETVAL_ADDR:.*]] = alloca i32
// LLVM:   store ptr %[[IC_ARG]], ptr %[[IC_ADDR]]
// LLVM:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// LLVM:   %[[IC:.*]] = load ptr, ptr %[[IC_ADDR]]
// LLVM:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// LLVM:   %[[RT_MEMBER:.*]] = getelementptr i8, ptr %[[IC]], i64 %[[MEMBER]]
// LLVM:   %[[VAL:.*]] = load i32, ptr %[[RT_MEMBER]]
// LLVM:   store i32 %[[VAL]], ptr %[[RETVAL_ADDR]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL_ADDR]]
// LLVM:   ret i32 %[[RET]]

// OGCG: define {{.*}} i32 @_Z5test5P10IncompleteMS_i(ptr {{.*}} %[[IC_ARG:.*]], i64 %[[MEMBER_ARG:.*]])
// OGCG:   %[[IC_ADDR:.*]] = alloca ptr
// OGCG:   %[[MEMBER_ADDR:.*]] = alloca i64
// OGCG:   store ptr %[[IC_ARG]], ptr %[[IC_ADDR]]
// OGCG:   store i64 %[[MEMBER_ARG]], ptr %[[MEMBER_ADDR]]
// OGCG:   %[[IC:.*]] = load ptr, ptr %[[IC_ADDR]]
// OGCG:   %[[MEMBER:.*]] = load i64, ptr %[[MEMBER_ADDR]]
// OGCG:   %[[RT_MEMBER:.*]] = getelementptr inbounds i8, ptr %[[IC]], i64 %[[MEMBER]]
// OGCG:   %[[RET:.*]] = load i32, ptr %[[RT_MEMBER]]
// OGCG:   ret i32 %[[RET]]

auto test_null() -> int Point::* {
  return nullptr;
}

// CIR: cir.func {{.*}} @_Z9test_nullv() -> !cir.data_member<!s32i in !rec_Point> {
// CIR:   %[[RETVAL_ADDR:.*]] = cir.alloca !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>, ["__retval"]
// CIR:   %[[CONST_NULL:.*]] = cir.const #cir.data_member<null> : !cir.data_member<!s32i in !rec_Point>
// CIR:   cir.store %[[CONST_NULL]], %[[RETVAL_ADDR]]
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL_ADDR]]
// CIR:   cir.return %[[RET]] : !cir.data_member<!s32i in !rec_Point>

// LLVM: define {{.*}} i64 @_Z9test_nullv()
// LLVM:   %[[RETVAL_ADDR:.*]] = alloca i64
// LLVM:   store i64 -1, ptr %[[RETVAL_ADDR]]
// LLVM:   %[[RET:.*]] = load i64, ptr %[[RETVAL_ADDR]]
// LLVM:   ret i64 %[[RET]]

// OGCG: define {{.*}} i64 @_Z9test_nullv()
// OGCG:   ret i64 -1

auto test_null_incomplete() -> int Incomplete::* {
  return nullptr;
}

// CIR: cir.func {{.*}} @_Z20test_null_incompletev() -> !cir.data_member<!s32i in !rec_Incomplete> {
// CIR:   %[[RETVAL_ADDR:.*]] = cir.alloca !cir.data_member<!s32i in !rec_Incomplete>, !cir.ptr<!cir.data_member<!s32i in !rec_Incomplete>>, ["__retval"]
// CIR:   %[[CONST_NULL:.*]] = cir.const #cir.data_member<null> : !cir.data_member<!s32i in !rec_Incomplete>
// CIR:   cir.store %[[CONST_NULL]], %[[RETVAL_ADDR]]
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL_ADDR]]
// CIR:   cir.return %[[RET]] : !cir.data_member<!s32i in !rec_Incomplete>

// LLVM: define {{.*}} i64 @_Z20test_null_incompletev()
// LLVM:   %[[RETVAL_ADDR:.*]] = alloca i64
// LLVM:   store i64 -1, ptr %[[RETVAL_ADDR]]
// LLVM:   %[[RET:.*]] = load i64, ptr %[[RETVAL_ADDR]]
// LLVM:   ret i64 %[[RET]]

// OGCG: define {{.*}} i64 @_Z20test_null_incompletev()
// OGCG:   ret i64 -1
