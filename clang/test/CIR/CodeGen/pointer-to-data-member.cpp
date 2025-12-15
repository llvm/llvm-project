// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
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
// CIR: cir.global external @pt_member = #cir.data_member<2> : !cir.data_member<!s32i in !rec_Point>
// LLVM: @pt_member = global i64 8
// OGCG: @pt_member = global i64 8

auto test1() -> int Point::* {
  return &Point::y;
}

// CIR: cir.func {{.*}} @_Z5test1v() -> !cir.data_member<!s32i in !rec_Point> {
// CIR:   %[[RETVAL:.*]] = cir.alloca !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>, ["__retval"]
// CIR:   %[[MEMBER:.*]] = cir.const #cir.data_member<1> : !cir.data_member<!s32i in !rec_Point>
// CIR:   cir.store %[[MEMBER]], %[[RETVAL]] : !cir.data_member<!s32i in !rec_Point>, !cir.ptr<!cir.data_member<!s32i in !rec_Point>>
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!cir.data_member<!s32i in !rec_Point>>, !cir.data_member<!s32i in !rec_Point>
// CIR:   cir.return %[[RET]] : !cir.data_member<!s32i in !rec_Point>

// LLVM: define {{.*}} i64 @_Z5test1v()
// LLVM:   %[[RETVAL:.*]] = alloca i64
// LLVM:   store i64 4, ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// LLVM:   ret i64 %[[RET]]

// OGCG: define {{.*}} i64 @_Z5test1v()
// OGCG:   ret i64 4

int test2(const Point &pt, int Point::*member) {
  return pt.*member;
}

// CIR:       cir.func {{.*}} @_Z5test2RK5PointMS_i(
// CIR-SAME:        %[[PT_ARG:.*]]: !cir.ptr<!rec_Point>
// CIR-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Point>
// CIR:         %[[PT_ADDR:.*]] = cir.alloca {{.*}} ["pt", init, const]
// CIR:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR:         cir.store %[[PT_ARG]], %[[PT_ADDR]]
// CIR:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR:         %[[PT:.*]] = cir.load %[[PT_ADDR]]
// CIR:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR:         %[[RT_MEMBER:.*]] = cir.get_runtime_member %[[PT]][%[[MEMBER]] : !cir.data_member<!s32i in !rec_Point>] : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CIR:         %[[VAL:.*]] = cir.load{{.*}} %[[RT_MEMBER]]
// CIR:         cir.store %[[VAL]], %[[RETVAL_ADDR]]
// CIR:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR:         cir.return %[[RET]]

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

// CIR:       cir.func {{.*}} @_Z5test3PK5PointMS_i(
// CIR-SAME:        %[[PT_ARG:.*]]: !cir.ptr<!rec_Point>
// CIR-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Point>
// CIR:         %[[PT_ADDR:.*]] = cir.alloca {{.*}} ["pt", init]
// CIR:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR:         cir.store %[[PT_ARG]], %[[PT_ADDR]]
// CIR:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR:         %[[PT:.*]] = cir.load{{.*}} %[[PT_ADDR]]
// CIR:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR:         %[[RT_MEMBER:.*]] = cir.get_runtime_member %[[PT]][%[[MEMBER]] : !cir.data_member<!s32i in !rec_Point>] : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CIR:         %[[VAL:.*]] = cir.load{{.*}} %[[RT_MEMBER]]
// CIR:         cir.store %[[VAL]], %[[RETVAL_ADDR]]
// CIR:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR:         cir.return %[[RET]]

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

// CIR:       cir.func {{.*}} @_Z5test4M10Incompletei(
// CIR-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Incomplete>
// CIR:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR:         cir.store %[[MEMBER]], %[[RETVAL_ADDR]]
// CIR:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR:         cir.return %[[RET]]

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

// CIR:       cir.func {{.*}} @_Z5test5P10IncompleteMS_i(
// CIR-SAME:        %[[IC_ARG:.*]]: !cir.ptr<!rec_Incomplete>
// CIR-SAME:        %[[MEMBER_ARG:.*]]: !cir.data_member<!s32i in !rec_Incomplete>
// CIR:         %[[IC_ADDR:.*]] = cir.alloca {{.*}} ["ic", init]
// CIR:         %[[MEMBER_ADDR:.*]] = cir.alloca {{.*}} ["member", init]
// CIR:         %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR:         cir.store %[[IC_ARG]], %[[IC_ADDR]]
// CIR:         cir.store %[[MEMBER_ARG]], %[[MEMBER_ADDR]]
// CIR:         %[[IC:.*]] = cir.load{{.*}} %[[IC_ADDR]]
// CIR:         %[[MEMBER:.*]] = cir.load{{.*}} %[[MEMBER_ADDR]]
// CIR:         %[[RT_MEMBER:.*]] = cir.get_runtime_member %[[IC]][%[[MEMBER]] : !cir.data_member<!s32i in !rec_Incomplete>] : !cir.ptr<!rec_Incomplete> -> !cir.ptr<!s32i>
// CIR:         %[[VAL:.*]] = cir.load{{.*}} %[[RT_MEMBER]]
// CIR:         cir.store %[[VAL]], %[[RETVAL_ADDR]]
// CIR:         %[[RET:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR:         cir.return %[[RET]]

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
